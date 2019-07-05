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

view_tracker::view_tracker() : classifier(cv::ml::NormalBayesClassifier::create()) {}

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

    z_range = (min - max) / 2;

    train_occlusion_classifier(mask, m_window_z);
}

void view_tracker::train_occlusion_classifier(cv::Mat mask, float depth) {
    cv::Mat dimg = m_view.depth()(m_window);

    dimg = m_view.disparity_to_depth(dimg) - depth;
    mask = mask(m_window);

    cv::Mat labels;

    mask.clone().convertTo(labels, CV_32S);

    classifier->train(dimg.clone().reshape(0, dimg.total()),  cv::ml::ROW_SAMPLE, labels.reshape(0, labels.total()));
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

float view_tracker::track(cv::Point3f predicted) {
    // Backproject histogram onto colour image
    cv::Mat pimg = backproject();

#ifdef FYP_TRACKER_2D
    // Perform 2D mean shift
    cv::meanShift(pimg, window, cv::TermCriteria(cv::TermCriteria::EPS | cv::TermCriteria::COUNT, 10, 1));
    return 0;

#else

    float weight = compute_area_covered();

    if (!is_occluded(predicted)) {
        // Perform 3D mean-shift
        std::tie(m_window, m_window_z) = mean_shift(pimg, m_view, m_window, m_window_z, 10, 1e-6, h);

        return weight;
    }

    return 0;
#endif
}

float view_tracker::compute_area_covered() {
    auto size = m_view.depth().size();
    cv::Rect r(std::min(std::max(0, m_window.x - m_window.width/2), size.width - m_window.width*2),
               std::min(std::max(0, m_window.y - m_window.height/2), size.height - m_window.height*2),
               m_window.width*2, m_window.height*2);

    cv::Mat img = m_view.disparity_to_depth(m_view.depth()(r));

//    cv::threshold(img, img, m_window_z - z_range, 1, cv::THRESH_TOZERO);
    cv::threshold(img, img, m_window_z + z_range, 1, cv::THRESH_TOZERO_INV);

    return cv::countNonZero(img);
}

int view_tracker::is_occluded(cv::Point3f predicted) {
    cv::Vec4f p(predicted.x, predicted.y, predicted.z, 1);

    // Transform predicted world space position to pixel space
    p = m_view.world_to_pixel(p);

    // Clamp tracking window to be on-screen

    auto size = m_view.depth().size();
    cv::Rect r(std::min(std::max(int(p[0]) - m_window.width / 2, 0), size.width - m_window.width),
               std::min(std::max(int(p[1]) - m_window.height / 2, 0), size.height - m_window.height),
               m_window.width, m_window.height);

    // Compute distances from predicted depth
    cv::Mat img = m_view.disparity_to_depth(m_view.depth()) - p[2];

    // Compute mean depth
    float mean = cv::mean(img(r))[0];

    cv::Mat m = cv::Mat(1,1, CV_32FC1);
    m.at<float>(0,0) = mean;


    // Determine whether mean depth value is that of the object
    auto out = classifier->predict(m);


    // If occluded set tracking window and depth to predicted values
    if (!out) {
        m_window = r;
        m_window_z = p[2];

        return true;
    }

    return false;
}

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

        // Number of rows and columns
        int rows = std::min(window.y + window.height, pimg.rows);
        int cols = std::min(window.x + window.width, pimg.cols);

#ifdef FYP_TRACKER_PARALLEL
#pragma omp parallel for reduction(+ : weights, sum_x, sum_y, sum_z)
#endif
        for (int y = window.y; y < rows; y++) {
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
            for (int x = window.x; x < cols; x++) {
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

            // Compute perspective projection to to obtain the
            // coordinates of the pixel at which the tracking window's
            // centre is located. Shift tracking window's centre to
            // the new centre.

            window.x = centre[0] - window.width / 2;
            window.y = centre[1] - window.height / 2;

            // Terminate loop if shift is less than the minimum
            if (distance < eps) break;
        }
    }

    // Return new tracking window and depth
    return std::make_pair(window, pos[2]);
}
