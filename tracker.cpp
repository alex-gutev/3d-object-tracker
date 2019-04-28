/*
 * Copyright (C) 2018  Alexander Gutev <alex.gutev@gmail.com>
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

#include <algorithm>
#include <numeric>

#include <valarray>
#include <iostream>

#include "tracker.h"


/// Reading and intialization

bool tracker::read() {
    for (view &v : views) {
        if (!v.read())
            return false;
    }

    return true;
}

view &tracker::add_view() {
    views.emplace_back();
    m_masks.emplace_back();
    m_hists.emplace_back();

    window_z.emplace_back();
    window.emplace_back();

    times.emplace_back();

    return views.back();
}

void tracker::primary_view(int v) {
    std::cout << v << std::endl;
    track_view = v;
}


/// Segmentation

/**
 * Returns the @a percent percentile of the values in the image,
 * within the mask.
 *
 * @param img       The image
 * @param percent   The percentile
 * @param mask      The mask of the pixels which are considered
 *
 * @return The percentile value
 */
static double percentile(cv::Mat img, double percent, cv::Mat mask) {
    int channels[] = {0};
    int histSize[] = {256};
    float hranges[] = {0, 255};
    const float *ranges[] = {hranges};

    cv::Mat hist;

    // Calculate histogram
    cv::calcHist(&img, 1, channels, mask, hist, 1, histSize, ranges);

    int total = 0;

    // Calculate the number of the pixel corresponding to the percentile
    double percentile = std::accumulate(hist.begin<float>(), hist.end<float>(), 0) * percent;

    int i = 0;
    for (auto it = hist.begin<float>(), end = hist.end<float>(); it != end; ++it) {
        total += *it;

        // If the number of pixels exceeds the percentile exit loop
        if (total >= percentile) {
            break;
        }

        i++;
    }

    // Return the value corresponding to the percentile
    return i;
}

int tracker::detect_objects(size_t view_index) {
    view &v = views[view_index];

    // Get depth image within window
    cv::Mat depth = v.depth()(window[view_index]);

    // Create mask image
    m_masks[view_index].create(v.depth().size(), CV_8UC1);
    m_masks[view_index] = cv::Scalar(0);

    // Get reference to mask image within window
    cv::Mat mask = m_masks[view_index](window[view_index]);

    // Reduce noise with a 3x3 kernel
    cv::blur(depth, mask, cv::Size(3,3));

    // Threshold depth image
    int threshold = cv::threshold(mask, mask, 0, 255, cv::THRESH_OTSU);

    // Map contours to remaining views
    map_regions(view_index, threshold);

    // Get z-range of object between the 66th and 33rd percentiles
    // Percentiles are used to prevent the range being skewed heavily
    // by noise.
    z_range = (v.disparity_to_depth(percentile(depth, 0.66, mask)) - v.disparity_to_depth(percentile(depth, 0.33, mask))) / 2;

    // Print position and size of each view's tracking window
    print_windows();

    return threshold;
}

void tracker::map_regions(size_t index, int depth_threshold) {
    view &src = views[index];

    std::vector<std::vector<cv::Point>> contours;

    // Find contours of object region
    cv::findContours(m_masks[index], contours, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);

    // Map contours to remaining views
    for (int i = 0; i < views.size(); i++) {
        if (i != index) {
            map_contours(contours, index, i, depth_threshold);
        }
    }
}

void tracker::map_contours(const contours_type &contours, size_t src, size_t dest, int depth_threshold) {
    view &v_src = views[src];
    view &v_dest = views[dest];

    // Create mask image for view
    m_masks[dest].create(v_dest.depth().size(), CV_8UC1);
    cv::Mat out_mask = m_masks[dest];

    out_mask = cv::Scalar::all(0);

    // Map contours to destination view
    for (auto &contour : contours) {
        contour_type new_contour;

        map_contour(contour, new_contour, v_src, v_dest);
        cv::drawContours(out_mask, contours_type({new_contour}), 0, cv::Scalar(255), cv::FILLED);
    }

    // Use morphology to remove noise in mapped contours
    cv::Mat structuringElement = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5,5));
    cv::morphologyEx(out_mask, out_mask, cv::MORPH_CLOSE, structuringElement);

    // Find all contour pixels
    std::vector<cv::Point> mask_points;
    cv::findNonZero(out_mask, mask_points);

    // Get bounding rectangle of contours
    cv::Rect bounds = cv::boundingRect(mask_points);

    out_mask = cv::Scalar::all(0);

    bounds.x -= map_region;
    bounds.y -= map_region;
    bounds.width += map_region;
    bounds.height += map_region;

    // Perform thresholding within bounding rectangle to determine
    // object's region
    cv::threshold(v_dest.depth()(bounds),
                  m_masks[dest](bounds),
                  depth_threshold,
                  255, cv::THRESH_BINARY);


    cv::Mat pts;
    cv::findNonZero(m_masks[dest](bounds), pts);

    // Get bounding rectangle of region
    cv::Rect rect = cv::boundingRect(pts);

    rect.x += bounds.x;
    rect.y += bounds.y;

    // Set bounding rectangle of region as view's initial tracking
    // window
    window[dest] = rect;
}


void tracker::map_contour(const contour_type &in_contour, contour_type &out_contour, view &src, view &dest) {
    // Map each contour pixel
    for (cv::Point pt : in_contour) {
        out_contour.push_back(src.transform(pt, dest));
    }
}


/// Colour Histogram

void tracker::build_models() {
    int channels[] = {0};
    int histSize[] = {180};
    float hranges[] = {0, 180};
    const float *ranges[] ={hranges};


    // Calculate hue histogram of each view

    for (int i = 0; i < views.size(); i++) {
        cv::Mat mask = m_masks[i];
        cv::Mat img;

        color(i).copyTo(img, mask);
        cv::cvtColor(img, img, cv::COLOR_BGR2HSV);

        cv::calcHist(&img, 1, channels, mask, m_hists[i], 1, histSize, ranges);
        cv::normalize(m_hists[i], m_hists[i], 0, 255, cv::NORM_MINMAX);

        window_z[i] = views[i].disparity_to_depth(cv::mean(depth(i), mask)[0]);
    }
}

void tracker::estimate_bandwidth() {
    view &v = views[track_view];

    cv::Rect region = window[track_view];
    float z = window_z[track_view];

    // Top-left corner of tracking window
    cv::Vec4f pt1 = cv::Vec4f(region.x * z, region.y * z, z, 1);
    // Centre of tracking window
    cv::Vec4f pt2 = cv::Vec4f((region.x + region.width / 2) * z, (region.y + region.height / 2) * z, z, 1);

    // Transform points to camera space
    pt1 = v.inv_intrinsic_matrix() * pt1;
    pt2 = v.inv_intrinsic_matrix() * pt2;

    // Compute distance between top-left and centre in camera space
    float dist = cv::norm(pt1 - pt2);

    // Compute bandwidth = median of distance and depth-range
    bandwidth = (dist + std::abs(z_range)) / 2;

}

cv::Mat tracker::backproject(size_t index) {
    view &v = views[index];

    cv::Mat hsv;
    cv::cvtColor(v.color(), hsv, cv::COLOR_BGR2HSV);

    cv::Mat dst;

    // Hue histogram

    int channels[] = {0};
    float hranges[] = {0, 180};
    const float *ranges[] = {hranges};

    cv::calcBackProject(&hsv, 1, channels, m_hists[index], dst, ranges);

    return dst;
}


/// Tracking

void tracker::track() {
#ifdef FYP_TRACKER_2D
    for (int i = 0; i < window.size(); i++) {
        track(i);
    }
#else
    // Track object in primary view
    track(track_view);

    // Primary view's tracking window
    cv::Rect rect = window[track_view];

    // Position of object in primary view
    cv::Point pos(rect.x + rect.width/2, rect.y + rect.height/2);

    // Z-coordinate of object's position in primary view
    float depth = window_z[track_view];

    view &primary_view = views[track_view];

    // For each of the remaining views
    for (int i = 0; i < views.size(); i++) {
        if (i != track_view) {
            // Map object's position in primary view to this view
            cv::Point3f new_pos = primary_view.transform(pos, depth, views[i]);

            // Shift z-coordinate of window
            window_z[i] = new_pos.z;

            cv::Rect rect = window[i];

            // Shift view's tracking to be centred at the mapped object position
            window[i].x = new_pos.x - rect.width / 2;
            window[i].y = new_pos.y - rect.height / 2;

            // Track the object in this view
            track(i);
        }
    }
#endif

    frames++;
}

void tracker::track(size_t index) {
    view &v = views[index];

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    // Backproject histogram onto colour image
    cv::Mat prob = backproject(index);

#ifdef FYP_TRACKER_2D
    // Perform 2D mean shift
    cv::meanShift(prob, window[index], cv::TermCriteria(cv::TermCriteria::EPS | cv::TermCriteria::COUNT, 10, 1));
#else
    // Perform 3D mean-shift
    std::tie(window[index], window_z[index]) = mean_shift(prob, v, window[index], window_z[index], 10, 1e-6, bandwidth);
#endif

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

    // Compute time taken by algorithm, in this frame, in milliseconds
    int milsecs = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();

    // Update per-frame average execution time using moving average formula
    times[index] = (milsecs + frames * times[index]) / (frames + 1);

    // Print object's position in this view
    print_position(index);
}

/**
 * Computes the magnitude of a vector.
 *
 * @param v The vector
 *
 * @return The magnitude
 */
template <typename T>
static float magnitude(T v) {
    float mag = 0;

    for (int i = 0; i < v.cols; i++) {
        mag += v[i] * v[i];
    }

    return sqrtf(mag);
}

/**
 * Computes the magnitude of a 3D vector
 *
 * @param pt The vector
 * @return The magnitude
 */
static float magnitude(cv::Point3f pt) {
    return sqrtf(pt.x * pt.x + pt.y * pt.y + pt.z * pt.z);
}

std::pair<cv::Rect, float> tracker::mean_shift(cv::Mat prob_img, view &v, cv::Rect window, float depth, int num_iters,
                                               float epsilon,
                                               float h) {
    // Position of object in 3D space
    cv::Vec4f pos((window.x + window.width/2) * depth, (window.y + window.height/2) * depth, depth, 1);
    // Reverse-project position to 3D camera space
    pos = v.inv_intrinsic_matrix() * pos;

    while (num_iters--) {
        // Sum of weights
        float weights = 0;

        // Total sum of each component
        float sum_x = 0;
        float sum_y = 0;
        float sum_z = 0;

        // Number of rows and columns
        int rows = std::min(window.y + window.height, prob_img.rows);
        int cols = std::min(window.x + window.width, prob_img.cols);

#ifdef FYP_TRACKER_PARALLEL
#pragma omp parallel for reduction(+ : weights, sum_x, sum_y, sum_z)
#endif
        for (int y = window.y; y < rows; y++) {
            // Pointer to the current row of the probability image
            const uchar *prob_row = prob_img.ptr(y);
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
            for (int x = window.x; x < cols; x++) {
                // Get probability
                float prob = prob_row[x] / 255.0f;

                // Get pixel's depth
                float z = v.disparity_to_depth(depth_row[x]);

                // Reverse perspective projection
                cv::Vec4f pt(x * z, y * z, z, 1);

                // Transform to camera space
                pt = v.inv_intrinsic_matrix() * pt;

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

            // Multiply by intrinsic matrix to convert to pixel space
            cv::Vec4f centre(v.intrinsic_matrix() * pos);

            // Compute perspective projection to to obtain the
            // coordinates of the pixel at which the tracking window's
            // centre is located. Shift tracking window's centre to
            // the new centre.

            window.x = centre[0]/centre[2] - window.width / 2;
            window.y = centre[1]/centre[2] - window.height / 2;

            // Terminate loop if shift is less than the minimum
            if (distance < epsilon) break;
        }
    }

    // Return new tracking window and depth
    return std::make_pair(window, pos[2]);
}

void tracker::print_position(int view) const {
    cv::Point tl = window[view].tl();

    std::cout << view << " " << tl.x << " " << tl.y << std::endl;
}

void tracker::print_windows() const {
    for (const cv::Rect &rect : window) {
        std::cout << rect.x << " " << rect.y << " ";
        std::cout << rect.width << " " << rect.height << std::endl;
    }
}

void tracker::print_time_stats() {
    for (int time : times) {
        std::cerr << time << std::endl;
    }
}
