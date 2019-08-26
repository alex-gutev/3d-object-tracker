/*
 * Copyright (C) 2018-2019  Alexander Gutev <alex.gutev@gmail.com>
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
#include "util.h"

/// Reading and intialization

bool tracker::read() {
    for (auto &tracker : trackers) {
        if (!tracker.view_info().read())
            return false;
    }

    return true;
}

view &tracker::add_view() {
    trackers.emplace_back();
    m_masks.emplace_back();

    return trackers.back().view_info();
}

void tracker::primary_view(int v) {
    track_view = v;
}


/// Segmentation

/**
 * Segments the depth, @a depth, and color, @a color, images by
 * watershedding. The results are combined into a single segmented
 * image by a bitwise and operation.
 *
 * @param mask Output matrix in which the object mask image is stored.
 * @param depth Depth image.
 * @param color Color image.
 */
static void get_mask(cv::Mat &mask, cv::Mat depth, cv::Mat color) {
    cv::Mat wcolor;

    watershed(depth, color, wcolor);

    cv::Mat cdepth;
    cv::Mat wdepth;
    cv::cvtColor(depth, cdepth, CV_GRAY2BGR);

    watershed(depth, cdepth, wdepth);

    cv::bitwise_and(wcolor, wdepth, mask);
}

/**
 * Determines the mask belonging indicating the pixels belonging to
 * the object in the region @a r.
 *
 * @param mask Output matrix in which object mask is stored.
 * @param r Region to segment.
 * @param depth Depth image.
 * @param color Color image.
 */
static void get_mask(cv::Mat &mask, cv::Rect r, cv::Mat depth, cv::Mat color) {
    get_mask(mask, depth, color);

    int l = mask.at<int>(r.y + r.height/2, r.x + r.width/2);

    mask.setTo(0, mask != l);
    mask.convertTo(mask, CV_8U);

    cv::Mat ones = cv::Mat::zeros(mask.size(), CV_8UC1);
    ones(r) = 255;

    cv::bitwise_and(mask, ones, mask);
    mask.setTo(255, mask);
}

void tracker::detect_objects(size_t view_index) {
    view_tracker &tracker = trackers[view_index];
    view &v = tracker.view_info();

    cv::Mat depth = v.depth().clone();
    cv::medianBlur(v.depth(), depth, 3);

    // Segment region within tracking window.
    get_mask(m_masks[view_index], tracker.window(), depth, v.color());

    cv::Mat mask = m_masks[view_index];

    cv::Mat pts;
    cv::findNonZero(mask, pts);
    tracker.window(cv::boundingRect(pts));

    // Map contours to remaining views
    map_regions(view_index);
}

void tracker::map_regions(size_t index) {
    std::vector<std::vector<cv::Point>> contours;

    // Find contours of object region
    cv::findContours(m_masks[index], contours, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);

    // Map contours to remaining views
    for (int i = 0; i < trackers.size(); i++) {
        if (i != index) {
            map_contours(contours, index, i);
        }
    }
}

void tracker::map_contours(const contours_type &contours, size_t src, size_t dest) {
    view &v_src = trackers[src].view_info();
    view &v_dest = trackers[dest].view_info();

    // Create image in which contours are drawn
    cv::Mat contour_image = cv::Mat::zeros(v_dest.depth().size(), CV_8UC1);

    // Map contours to destination view
    for (auto &contour : contours) {
        contour_type new_contour;

        map_contour(contour, new_contour, v_src, v_dest);
        cv::drawContours(contour_image, contours_type({new_contour}), 0, cv::Scalar(255), cv::FILLED);
    }

    // Use morphology to remove noise in mapped contours
    cv::Mat structuringElement = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5,5));
    cv::morphologyEx(contour_image, contour_image, cv::MORPH_CLOSE, structuringElement);

    // Find all contour pixels
    std::vector<cv::Point> mask_points;
    cv::findNonZero(contour_image, mask_points);

    // Get bounding rectangle of contours to get segmentation region
    cv::Rect bounds = cv::boundingRect(mask_points);

    // Clamp bounding rectange to be on screen
    bounds.x = clamp(bounds.x - map_region, 0, contour_image.size().width - 1);
    bounds.y = clamp(bounds.y - map_region, 0, contour_image.size().height - 1);
    bounds.width = clamp(bounds.width + map_region, 0, contour_image.size().width - bounds.x);
    bounds.height = clamp(bounds.height + map_region, 0, contour_image.size().height - bounds.y);

    // Get object mask in view
    get_mask(m_masks[dest], bounds, v_dest.depth(), v_dest.color());


    // Get points making up region
    cv::Mat pts;
    cv::findNonZero(m_masks[dest], pts);

    // Get bounding rectangle of region
    cv::Rect rect = cv::boundingRect(pts);

    // Set bounding rectangle of region as view's initial tracking
    // window
    trackers[dest].window(rect);
}


void tracker::map_contour(const contour_type &in_contour, contour_type &out_contour, view &src, view &dest) {
    // Map each contour pixel
    for (cv::Point pt : in_contour) {
        cv::Point t = src.transform(pt, dest);

        // Clamp point to be within depth map image
        t.x = clamp(t.x, 0, dest.depth().size().width - 1);
        t.y = clamp(t.y, 0, dest.depth().size().height - 1);

        out_contour.push_back(t);
    }
}


/// Kalman Filtering

void tracker::init_kalman_filter() {
    // Initialize Transition Matrix

    kmfilter.transitionMatrix = cv::Mat::eye(6, 6, CV_32F);
    kmfilter.transitionMatrix.at<float>(0,3) = 1;
    kmfilter.transitionMatrix.at<float>(1,4) = 1;
    kmfilter.transitionMatrix.at<float>(2,5) = 1;

    // Initialize Measurement Matrix

    kmfilter.measurementMatrix = cv::Mat::eye(6, 6, CV_32F);

    // Initialize Process and Measurement Noise Covariance Matrices

    kmfilter.processNoiseCov = 1e-5 * cv::Mat::eye(6, 6, CV_32F);
    kmfilter.measurementNoiseCov = 1e-1 * cv::Mat::eye(6, 6, CV_32F);
}

void tracker::init_kalman_state(float x, float y, float z) {
    kmfilter.statePost = cv::Mat::zeros(6, 1, CV_32F);

    kmfilter.statePost.at<float>(0) = x;
    kmfilter.statePost.at<float>(1) = y;
    kmfilter.statePost.at<float>(2) = z;

    kmfilter.errorCovPost = 0.1 * cv::Mat::eye(6, 6, CV_32F);
}

cv::Mat tracker::kalman_predict() {
    return kmfilter.predict();
}

cv::Mat tracker::kalman_correct(int view) {
    auto &tracker = trackers[view];
    cv::Rect r = tracker.window();
    float z = tracker.window_z();

    cv::Vec4f c = tracker.view_info().pixel_to_world(r.x + r.width/2, r.y + r.height/2, z);

    return kalman_correct(c[0], c[1], c[2]);
}

cv::Mat tracker::kalman_correct(float x, float y, float z) {
    cv::Mat m(6, 1, CV_32F);

    m.at<float>(0) = x;
    m.at<float>(1) = y;
    m.at<float>(2) = z;

    m.at<float>(3) = x - kmfilter.statePost.at<float>(0);
    m.at<float>(4) = y - kmfilter.statePost.at<float>(1);
    m.at<float>(5) = z - kmfilter.statePost.at<float>(2);

    return kmfilter.correct(m);
}


void tracker::update_windows() {
    cv::Vec4f pos({kmfilter.statePost.at<float>(0),
                   kmfilter.statePost.at<float>(1),
                   kmfilter.statePost.at<float>(2),
                   1});

    for (size_t i = 0; i < trackers.size(); ++i) {
        update_window(i, pos);
    }

}

void tracker::update_window(int view, cv::Vec4f pos) {
    auto &tracker = trackers[view];
    auto c = tracker.view_info().world_to_pixel(pos);

    cv::Rect w = tracker.window();

    w.x = c[0] - w.width/2;
    w.y = c[1] - w.height/2;

    tracker.window(w);
    tracker.window_z(c[2]);
}

/// Colour Histogram

void tracker::build_models() {
    int i = 0;
    for (auto &tracker : trackers) {
        tracker.build_model(m_masks[i]);
        i++;
    }
}

void tracker::estimate_bandwidth() {
    for (auto &tracker : trackers) {
        tracker.estimate_bandwidth();
    }

    init_kalman_filter();

    auto &tracker = trackers[track_view];
    auto r = tracker.window();
    auto p = tracker.view_info().pixel_to_world(r.x + r.width/2, r.y + r.height/2, tracker.window_z());

    init_kalman_state(p[0], p[1], p[2]);
}

/// Tracking

void tracker::track() {
#ifdef FYP_TRACKER_2D
    for (int i = 0; i < window.size(); i++) {
        track(i);
    }
#else
    kalman_predict();

    cv::Vec4f pos({0,0,0,0});
    float weights = 0;

    for (size_t i = 0; i < trackers.size(); ++i) {
        auto &tracker = trackers[i];
        float weight = track(i);

        cv::Rect r = tracker.window();
        float z = tracker.window_z();

        pos += weight * tracker.view_info().pixel_to_world(r.x + r.width/2, r.y + r.height/2, z);
        weights += weight;
    }

    if (weights) {
        pos /= weights;

        kalman_correct(pos[0], pos[1], pos[2]);

        update_windows();
    }

#endif
}

float tracker::track(size_t index) {
    float weight = trackers[index].track({
            kmfilter.statePre.at<float>(0),
            kmfilter.statePre.at<float>(1),
            kmfilter.statePre.at<float>(2)
        },
        {
            kmfilter.statePre.at<float>(3),
            kmfilter.statePre.at<float>(4),
            kmfilter.statePre.at<float>(5)
        });

    return weight;
}
