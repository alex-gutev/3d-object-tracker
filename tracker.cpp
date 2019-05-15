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

int tracker::detect_objects(size_t view_index) {
    view_tracker &tracker = trackers[view_index];
    view &v = tracker.view_info();

    // Get depth image within window
    cv::Mat depth = v.depth()(tracker.window());

    // Create mask image
    m_masks[view_index].create(v.depth().size(), CV_8UC1);
    m_masks[view_index] = cv::Scalar(0);

    // Get reference to mask image within window
    cv::Mat mask = m_masks[view_index](tracker.window());

    // Reduce noise with a 3x3 kernel
    cv::blur(depth, mask, cv::Size(3,3));

    // Threshold depth image
    int threshold = cv::threshold(mask, mask, 0, 255, cv::THRESH_OTSU);

    // Map contours to remaining views
    map_regions(view_index, threshold);

    return threshold;
}

void tracker::map_regions(size_t index, int depth_threshold) {
    std::vector<std::vector<cv::Point>> contours;

    // Find contours of object region
    cv::findContours(m_masks[index], contours, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);

    // Map contours to remaining views
    for (int i = 0; i < trackers.size(); i++) {
        if (i != index) {
            map_contours(contours, index, i, depth_threshold);
        }
    }
}

void tracker::map_contours(const contours_type &contours, size_t src, size_t dest, int depth_threshold) {
    view &v_src = trackers[src].view_info();
    view &v_dest = trackers[dest].view_info();

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
    trackers[dest].window(rect);
}


void tracker::map_contour(const contour_type &in_contour, contour_type &out_contour, view &src, view &dest) {
    // Map each contour pixel
    for (cv::Point pt : in_contour) {
        out_contour.push_back(src.transform(pt, dest));
    }
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
    cv::Rect rect = trackers[track_view].window();

    // Position of object in primary view
    cv::Point pos(rect.x + rect.width/2, rect.y + rect.height/2);

    // Z-coordinate of object's position in primary view
    float depth = trackers[track_view].window_z();

    view &primary_view = get_view(track_view);

    // For each of the remaining views
    for (int i = 0; i < trackers.size(); i++) {
        if (i != track_view) {
            auto &tracker = trackers[i];

            // Map object's position in primary view to this view
            cv::Point3f new_pos = primary_view.transform(pos, depth, get_view(i));

            // Shift z-coordinate of window
            tracker.window_z(new_pos.z);

            cv::Rect rect = tracker.window();

            // Shift view's tracking to be centred at the mapped object position
            rect.x = new_pos.x - rect.width / 2;
            rect.y = new_pos.y - rect.height / 2;

            tracker.window(rect);

            // Track the object in this view
            track(i);
        }
    }
#endif
}

void tracker::track(size_t index) {
    trackers[index].track();
}
