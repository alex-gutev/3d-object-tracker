/*
 * Copyright (C) 2019  Alexander Gutev <alex.gutev@gmail.com>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

#include "view_tracker.h"

#include <algorithm>
#include <numeric>

#include <opencv2/ximgproc.hpp>

#include "util.h"

// Initialization: Appearance and Depth Models ////////////////////////////////

void view_tracker::build_model(cv::Mat mask) {
    int channels[] = {0};
    int histSize[] = {180};
    float hranges[] = {0, 180};
    const float *ranges[] ={hranges};

    // Calculate Hue Histogram

    cv::Mat img;
    m_view.color().copyTo(img, mask);
    cv::cvtColor(img, img, cv::COLOR_BGR2HSV);

    cv::calcHist(&img, 1, channels, mask, hist, 1, histSize, ranges);
    cv::normalize(hist, hist, 0, 255, cv::NORM_MINMAX);

    m_window_z = m_view.disparity_to_depth(cv::mean(m_view.depth(), mask)[0]);


    // Estimate depth range

    // Get reference to mask image within window
    cv::Mat depth = m_view.depth()(m_window);

    cv::medianBlur(m_view.depth()(m_window), depth, 3);

    cv::Mat submask = mask(m_window);

    double min, max;

    cv::minMaxLoc(depth, &min, &max, nullptr, nullptr, submask);

    min = m_view.disparity_to_depth(min);
    max = m_view.disparity_to_depth(max);

    z_range = cv::abs(min - max) / 2;
}

void view_tracker::estimate_bandwidth() {
    // Top-left corner of tracking window
    cv::Vec4f p1(m_window.x * m_window_z, m_window.y * m_window_z, m_window_z, 1);

    // Centre of tracking window
    cv::Vec4f p2((m_window.x + m_window.width / 2) * m_window_z,
                 (m_window.y + m_window.height / 2) * m_window_z,
                 m_window_z,
                 1);

    // Transform points to camera space
    p1 = m_view.inv_intrinsic_matrix() * p1;
    p2 = m_view.inv_intrinsic_matrix() * p2;

    // Compute distance between top-left and centre in camera space
    float dist = cv::norm(p1 - p2);

    // Compute bandwidth = median of distance and depth-range
    h = (dist + std::abs(z_range)) / 2;
}

void view_tracker::bandwidth(float value) {
    h = value;
}


// Tracking ///////////////////////////////////////////////////////////////////

float view_tracker::track(cv::Point3f predicted, cv::Vec3f velocity) {
    // Backproject histogram onto colour image
    cv::Mat pimg = backproject();

#ifdef FYP_TRACKER_2D
    // Perform 2D mean shift
    cv::meanShift(pimg, window, cv::TermCriteria(cv::TermCriteria::EPS | cv::TermCriteria::COUNT, 10, 1));
    return 0;

#else

    float weight = compute_area_covered();

    cv::Rect new_window;
    float new_z;

    // Perform 3D mean-shift
    std::tie(new_window, new_z) = mean_shift(pimg, m_view, m_window, m_window_z, 10, 1e-6, h);

    if (!is_occluded(new_window, new_z, predicted)) {
        return weight;
    }

    cv::Vec4f pixel = m_view.world_to_pixel(cv::Vec4f(predicted.x, predicted.y, predicted.z));

    if (check_passed_occluder(cv::Point(pixel[0], pixel[1]), velocity)) {
        return weight;
    }

    m_window = cv::Rect(pixel[0] - m_window.width / 2, pixel[1] - m_window.height / 2,
                        m_window.width, m_window.height);
    m_window_z = pixel[2];

    return 0;
#endif
}

float view_tracker::compute_area_covered() {
    auto size = m_view.depth().size();

    cv::Rect r = clamp_region(
        cv::Rect(
            m_window.x - m_window.width / 2,
            m_window.y - m_window.height / 2,
            m_window.width *2,
            m_window.height *2
            ),
        size);

    cv::Mat img = m_view.disparity_to_depth(m_view.depth()(r));

//    cv::threshold(img, img, m_window_z - z_range, 1, cv::THRESH_TOZERO);
    cv::threshold(img, img, m_window_z + z_range, 1, cv::THRESH_TOZERO_INV);

    return cv::countNonZero(img);
}


// Occlusion Detection ////////////////////////////////////////////////////////

bool view_tracker::is_occluded(cv::Rect r, float z, cv::Point3f predicted) {
    cv::Vec4f p = m_view.world_to_pixel(cv::Vec4f(predicted.x, predicted.y, predicted.z));
    float pz = p[2];

    r.x = clamp(r.x, 0, m_view.depth().size().width - r.width);
    r.y = clamp(r.y, 0, m_view.depth().size().height - r.height);


    auto new_objects = detect_objects(r);



    for (auto &obj : new_objects) {
        // If the object's type has not already been determined,
        // i.e. it was not matched to an object in the previous frame.
        if (!obj.type) {
            if (obj.max < pz && obj.max < m_window_z) {
                obj.type = object::type_occluder;
            }
            else if (obj.min < pz && pz < obj.max) {
                obj.type = object::type_target;
            }
            else if (cv::abs(pz - obj.depth) < z_range) {
                obj.type = object::type_target;
            }
            else {
                obj.type = object::type_background;
            }
        }
    }

    bool occ = true;
    float new_z = 0;

    std::tie(occ, new_z) = is_occluded(new_objects, r, z);

    objects = std::move(new_objects);

    // True if background region detected
    bool detected_bg = false;
    // Distance to closest background region
    float min_bg = 0;

    for (auto &obj : objects) {
        if (obj.type == object::type_background) {
            if (!detected_bg || obj.min < min_bg) {
                detected_bg = true;
                min_bg = obj.min;
            }
        }
    }

    if (detected_bg)
        dist_background = min_bg;

    if (!occ) m_window_z = new_z;
    return occ;
}

std::pair<bool, float> view_tracker::is_occluded(const std::vector<object> &objects, cv::Rect window, float z) {
    bool occ = true;
    float new_z = 0;

    int closest = -1;
    float closest_dist = 0;

    size_t i = 0;

    cv::Mat points;

    for (auto &obj : objects) {
        // If object is an occluder and the z position found by
        // mean-shift lies within the object's z range, return true.
        if (obj.type == object::type_occluder) {
            if (obj.min < z && z < obj.max) {
                new_z = obj.max + z_range/4;
                occ = true;
                break;
            }
        }

        // If the object is part of the target, set new z-coordinate
        // to median depth of the object.
        if (obj.type == object::type_target) {
            cv::findNonZero(obj.region, points);

            float d = cv::abs(z - obj.depth);
            if ((obj.min < z && z < obj.max) ||
               (d < z_range)) {

                if (closest == -1 || d < closest_dist) {
                    closest = i;
                    closest_dist = d;
                }
                occ = false;
            }
        }

        i++;
    }

    if (!occ) {
        cv::Rect r = cv::boundingRect(points);

        m_window.x = (r.x + r.width/2) - window.width/2;
        m_window.y = (r.y + r.height/2) - window.height/2;

        // m_window = window;
        new_z = objects[closest].depth;
    }

    return std::make_pair(occ, new_z);
}


void view_tracker::merge_objects(cv::Mat img, std::vector<object> &objects) {
    // Sort in order of minimum depth value
    std::sort(objects.begin(), objects.end(), [=] (const object &a, const object & b) {
        return a.min < b.min;
    });

    // Last object that was retained
    object *old = nullptr;

    // Iterate through object array
    auto it = objects.begin();
    while (it != objects.end()) {
        auto &obj = *it;

        if (!old) {
            old = &*it;
        }
        else if (old->min <= obj.depth && obj.depth <= old->max) {
            // Merge regions
            old->region |= obj.region;

            // Recompute depth statistics
            old->min = percentile(img, 0.05, old->region);
            old->max = percentile(img, 0.95, old->region);
            old->depth = percentile(img, 0.5, old->region);

            // Remove current object
            it = objects.erase(it);
            continue;
        }
        else {
            old = &obj;
        }

        ++it;
    }
}

std::vector<view_tracker::object> view_tracker::detect_objects(cv::Rect r) const {
    // Convert disparity map to depth map
    cv::Mat dimg = m_view.disparity_to_depth(m_view.depth()(r));
    cv::Mat img;

    // Normalize depth map to range [0, 255] in order to perform MS clustering
    cv::normalize(dimg, img, 0, 255, cv::NORM_MINMAX, CV_8UC1);

    // Convert to 3-channel image for pyrMeanShiftFiltering function
    cv::Mat channels[] = {img, img, img};
    cv::merge(channels, 3, img);

    // Perform Mean Shift Clustering
    cv::pyrMeanShiftFiltering(img, img, 10, z_range);

    // Extract single channel from clustered image
    cv::Mat simg;
    cv::extractChannel(img, simg, 0);

    // Segment clustered image
    cv::Mat seg;
    size_t n = watershed(~simg, img, seg);

    // Compute statistics of new objects in scene
    std::vector<object> new_objects;

    for (int i = 1; i < n; ++i) {
        if (cv::countNonZero(seg == i)) {
            auto median = percentile(dimg, 0.5, seg == i);

            auto min = percentile(dimg, 0.5, seg == i);
            auto max = percentile(dimg, 0.95, seg == i);

            new_objects.emplace_back(object::type_unknown, min, max, median);
            new_objects.back().region = seg == i;

            cv::Mat points;
            cv::findNonZero(seg == i, points);
            cv::Rect box = cv::boundingRect(points);

            auto pt = m_view.pixel_to_world(box.x + box.width/2.0f, box.y + box.height / 2.0f, median);

            new_objects.back().pos = cv::Point3f(pt[0], pt[1], pt[2]);
            new_objects.back().bounds = box;
        }
    }

    match_objects(new_objects, r);

    return new_objects;
}

/**
 * Stores information about the similarity between an object in the
 * previous frame and current frame.
 */
struct matching {
    /**
     * 3D Euclidean distance between object centres.
     */
    float dist;

    /**
     * Index of object in previous frame.
     */
    size_t old;

    /**
     * Index of object in current frame.
     */
    size_t current;

    /**
     * Constructor
     *
     * @param dist 3D Euclidean distance between objects.
     * @param old Index of object in previous frame.
     * @param current Index of object in current frame.
     */
    matching(float dist, size_t old, size_t current) :
        dist(dist), old(old), current(current) {}
};

void view_tracker::match_objects(std::vector<object> &new_objects, cv::Rect r) const {
    std::vector<matching> pairs;

    // Compute distances between each pair of old and new objects.

    size_t new_i = 0;

    for (auto new_obj : new_objects) {
        size_t old_i = 0;

        for (auto old : objects) {
            float d = magnitude(old.pos - new_obj.pos);

            float old_area = cv::countNonZero(old.region);
            float new_area = cv::countNonZero(new_obj.region);
            float overlap_area = cv::countNonZero(old.region & new_obj.region);
            float overlap = overlap_area / new_area;

            // If there isn't significant overlap between the objects
            // then don't consider it a possible match

            if (overlap > 0.5) {
                pairs.emplace_back(d, old_i, new_i);
            }

            old_i++;
        }

        new_i++;
    }


    // Sort by distance in ascending order

    std::sort(pairs.begin(), pairs.end(), [](const matching &a, const matching &b) {
        return a.dist < b.dist;
    });

    // Iterate through each pair starting from pair with smallest
    // distance, while the array is not empty.
    while (pairs.size()) {
        auto &match = pairs.front();

        // Update type of new object to match old object
        new_objects[match.current].type = objects[match.old].type;

        // Remove remaining pairings involving either the current or
        // previous object.
        pairs.erase(
            std::remove_if(pairs.begin(), pairs.end(), [=] (const matching &m) {
                return m.old == match.old || m.current == match.current;
            }),
            pairs.end()
            );
    }
}


// Detecting Re-emergence /////////////////////////////////////////////////////

bool view_tracker::check_passed_occluder(cv::Point p, cv::Vec3f v) {
    // Previous object position in 3D world space
    cv::Vec4f p3 = m_view.pixel_to_world(m_window.x + m_window.width/2, m_window.y + m_window.height / 2, m_window_z);

    // Shift in x-y plane by x and y velocity components.
    // Ignore velocity in z axis.
    p3[0] += v[0];
    p3[1] += v[1];

    p3 = m_view.world_to_pixel(p3);


    // Predicted tracking window
    cv::Rect r(p3[0] - m_window.width/2, p3[1] - m_window.height/2, m_window.width, m_window.height);
    r = clamp_region(r, m_view.depth().size());


    // Segment region
    auto ed = cv::ximgproc::segmentation::createGraphSegmentation();

    cv::Mat seg;
    ed->processImage(m_view.depth()(r), seg);

    // Determine number of regions
    double max;
    cv::minMaxIdx(seg, nullptr, &max);
    size_t n = max + 1;

    // Compute statistics of new objects in scene
    std::vector<object> objs;

    cv::Mat dimg = m_view.disparity_to_depth(m_view.depth())(r);

    for (int i = 0; i < n; ++i) {
        cv::Mat region = seg == i;

        if (cv::countNonZero(region)) {
            auto median = percentile(dimg, 0.5, region);

            auto min = percentile(dimg, 0.05, region);
            auto max = percentile(dimg, 0.95, region);

            objs.emplace_back(object::type_unknown, min, max, median);
            objs.back().region = region;
        }
    }

    merge_objects(dimg, objs);

    // Flag: True if target object was found
    bool found_object = false;
    // Target object z position
    float new_z;
    // Target object region points
    cv::Mat obj_points;

    for (auto & obj : objs) {
        if ((obj.min < m_window_z && m_window_z < obj.max) ||
            cv::abs(m_window_z - obj.depth) < cv::abs(dist_background - obj.depth)) {

            cv::findNonZero(obj.region, obj_points);

            if (!found_object || obj.depth < new_z) {
                found_object = true;
                new_z = obj.depth;
            }
        }
    // If target region was found
    if (found_object) {
        m_window_z = new_z;

        cv::Rect newr = cv::boundingRect(obj_points);

        m_window.x = (r.x + newr.x + newr.width/2) - m_window.width/2;
        m_window.y = (r.y + newr.y + newr.height/2) - m_window.height/2;

        return true;
    }

    return false;
}


// Mean Shift Tracking ////////////////////////////////////////////////////////

cv::Mat view_tracker::backproject() {
    cv::Mat hsv;
    cv::cvtColor(m_view.color(), hsv, cv::COLOR_BGR2HSV);

    cv::Mat dst;

    // Hue histogram

    int channels[] = {0};
    float hranges[] = {0, 180};
    const float *ranges[] = {hranges};

    cv::calcBackProject(&hsv, 1, channels, hist, dst, ranges);

    return dst;
}

std::pair<cv::Rect, float> view_tracker::mean_shift(cv::Mat pimg, view &v, cv::Rect window, float depth, int num_iters, float eps, float h) {
    // Convert object position (center of tracking window) to camera space
    cv::Vec4f pos = v.pixel_to_camera(window.x + window.width / 2, window.y + window.height / 2, depth);

    while (num_iters--) {
        // Sum of weights
        float weights = 0;

        // Total sum of each component
        float sum_x = 0;
        float sum_y = 0;
        float sum_z = 0;

        // Rectangle clamped to visible image
        cv::Rect r = clamp_rect(window, pimg.size());

        // Number of rows and columns
        int rows = std::min(r.y + r.height, pimg.rows);
        int cols = std::min(r.x + r.width, pimg.cols);

#ifdef FYP_TRACKER_PARALLEL
#pragma omp parallel for reduction(+ : weights, sum_x, sum_y, sum_z)
#endif
        for (int y = r.y; y < rows; y++) {
            // Pointer to the current row of the probability image
            const uchar *prob_row = pimg.ptr(y);
            // Pointer to the current row of the depth image
            const uchar *depth_row = v.depth().ptr(y);

            // Row total weights
            float row_weights = 0;

            // Row total components
            float row_sum_x = 0;
            float row_sum_y = 0;
            float row_sum_z = 0;

#ifdef FYP_TRACKER_PARALLEL
#pragma omp parallel for reduction(+ : row_weights, row_sum_x, row_sum_y, row_sum_z)
#endif
            for (int x = r.x; x < cols; x++) {
                // Get probability
                float prob = prob_row[x] / 255.0f;

                // Get pixel's depth
                float z = v.disparity_to_depth(depth_row[x]);

                // Transform center of tracking window to camera space
                cv::Vec4f pt = v.pixel_to_camera(x, y, z);

                // Compute 3D Euclidean distance
                float dist = magnitude(cv::Point3f(pos[0] - pt[0], pos[1] - pt[1], pos[2] - pt[2]));

                // Compute weight = kernel weight * probability
                float weight = expf(-0.5f * powf(dist/h, 2)) * prob;

                // Add to row weight total
                row_weights += weight;

                // Add to row component totals
                row_sum_x += pt[0] * weight;
                row_sum_y += pt[1] * weight;
                row_sum_z += pt[2] * weight;
            }

            // Add row weight total to iteration weight total
            weights += row_weights;

            // Add row component totals to iteration component totals
            sum_x += row_sum_x;
            sum_y += row_sum_y;
            sum_z += row_sum_z;
        }

        // If weights is non-zero.  An all zero weight total can occur
        // if the probabilities of all pixels are very low and the
        // distances are very large
        if (weights) {

            // Compute new position using mean-shift formula
            cv::Vec4f new_pos(sum_x / weights, sum_y / weights, sum_z / weights, 1);

            // Compute distance between new and old position
            float distance = magnitude(pos - new_pos);

            // Update position
            pos = new_pos;

            // Convert to pixel space
            cv::Vec4f centre = v.camera_to_pixel(pos);


            window.x = centre[0] - window.width / 2;
            window.y = centre[1] - window.height / 2;

            // Terminate loop if shift is less than the minimum
            if (distance < eps) break;
        }
    }

    // Return new tracking window and depth
    return std::make_pair(window, pos[2]);
}
