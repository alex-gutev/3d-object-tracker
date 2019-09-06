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

#ifndef FYP_TRACKER_H
#define FYP_TRACKER_H

#include <vector>
#include <chrono>

#include <opencv2/imgproc.hpp>
#include <opencv2/video.hpp>

#include "view_tracker.h"

/**
 * Performs object detection and tracking in multi-view content.
 */
class tracker {
    /**
     * Number of pixels added to each side of the bounding rectangle
     * of the mapped contours. This is used to determine the region in
     * which thresholding is performed in the remaining views.
     */
    static constexpr int map_region = 10;

    /**
     * Single contour type.
     */
    typedef std::vector<cv::Point> contour_type;
    /**
     * Array of contours type.
     */
    typedef std::vector<contour_type> contours_type;

    /**
     * Array of trackers for each view.
     */
    std::vector<view_tracker> trackers;

    /**
     * Array of binary mask images, corresponding to the regions which
     * are occupied by the object in each view.
     *
     * These are used to compute the colour histogram in each view,
     * and are only valid for the frame in which object detection was
     * performed.
     */
    std::vector<cv::Mat> m_masks;

    /**
     * The primary view in which tracking is performed.  The position
     * obtained in this view is used to infer the starting position,
     * of the mean shift tracker, in the other views.
     */
    int track_view;


    /**
     * Kalman filter process noise.
     */
    static constexpr float process_noise = 1e-5;
    /**
     * Kalman filter measurement noise.
     */
    static constexpr float measurement_noise = 1e-1;

    /**
     * Kalman filter with six state variables:
     *
     * position: x,y,z
     * velocity: vx, vy, vz
     */
    cv::KalmanFilter kmfilter{6, 6};


    /** Object detection */

    /**
     * Maps the object region detected in a view (with index @a view)
     * to the corresponding object region in the rest of the views.
     *
     * The depth threshold is used to perform thresholding in the
     * views, within the mapped region.
     *
     * @param view  Index of the view in which the region was detected.
     */
    void map_regions(size_t view);

    /**
     * Maps contours from the view @a src to the view @a dest, and
     * performs depth thresholding in the region designated by the
     * mapped contours.
     *
     * @param contours          The contours to be mapped.
     * @param src               The source view.
     * @param dest              The destination view.
     */
    void map_contours(const contours_type &contours, size_t src, size_t dest);

    /**
     * Maps a single contour from the view @a src to the view @a dest
     *
     * @param in_contour        The countour to be mapped.
     * @param out_contour       The output array, where the mapped contour points will be stored.
     * @param src               The source view.
     * @param dest              The destination view.
     */
    void map_contour(const contour_type &in_contour, contour_type &out_contour, view &src, view &dest);


    /* Kalman Filtering */

    /**
     * Initialize Kalman Filter Matrices.
     */
    void init_kalman_filter();

    /**
     * Set initial Kalman Filter state.
     *
     * @param x X position.
     * @param y Y position.
     * @param z Z position.
     */
    void init_kalman_state(float x, float y, float z);

    /**
     * Predict object position using Kalman Filter.
     */
    cv::Mat kalman_predict();

    /**
     * Correct Kalman Filter state using the position of the tracking window
     * in view @a view, as a measurement.
     *
     * @param view The view whose tracking window to use as a
     * measurement.
     */
    cv::Mat kalman_correct(int view);

    /**
     * Correct Kalman Filter state using measured object position.
     *
     * @param x X position.
     * @param y Y position.
     * @param z Z position.
     */
    cv::Mat kalman_correct(float x, float y, float z);

    /**
     * Update the position of all tracking windows to the current
     * position of the Kalman Filter.
     */
    void update_windows();
    /**
     * Update the position of the tracking window in view @a view.
     *
     * @param view The view whose tracking window to update.
     * @param pos The new centre of the tracking window.
     */
    void update_window(int view, cv::Vec4f pos);

public:

    /**
     * Performs object detection within a view's tracking window.  The
     * object's contours are then mapped to the neighbouring views.
     *
     * After this function is called, a mask corresponding to the
     * region occuppied by the object is obtained, for each view.
     *
     * @param view Index of the view in which to perform object
     * detection.
     */
    void detect_objects(size_t view);

    /**
     * Builds the appearance models, for each view, of the object.
     */
    void build_models();

    /**
     * Estimates the kernel bandwidth
     */
    void estimate_bandwidth();


    /** Tracking */

    /**
     * Returns the primary view in which tracking is performed.
     *
     * Tracking in the primary view is performed first, after which
     * the new position is used to infer the starting position of the
     * tracker in the remaining views.
     *
     * @return The index of the view.
     */
    int primary_view() const {
        return track_view;
    }

    /**
     * Sets the primary view in which tracking is to be performed.
     *
     * @param v Index of the view.
     */
    void primary_view(int v);

    /**
     * Tracks the object in each view, starting in the primary view
     * and using the computed position to infer the starting position
     * of the tracker in the remaining views.
     */
    void track();

    /**
     * Tracks the object in a single view.
     *
     * @param index The index of the view.
     *
     * @return The amount, by which, the position computed by the
     *   tracker in this view should be weighted in merging the
     *   positions of each view.
     */
    float track(size_t index);


    /** IO and Initialization */

    /**
     * Reads a frame from the colour and depth video files of each
     * view.
     *
     * @return True if a frame was read successfully from each view's
     *         video files, false if EOF was reached in at least one
     *         video.
     */
    bool read();

    /**
     * Adds a new view object to the @a views array.
     *
     * @return Reference to the new view object.
     */
    view &add_view();


    /** Accessors */

    /**
     * Returns a reference to a view object.
     *
     * @param index Index of the view.
     * @return Reference to the view object.
     */
    view &get_view(size_t index) {
        return trackers.at(index).view_info();
    }

    /**
     * Returns the colour frame image of a view.
     *
     * @param view View index.
     * @return The frame image.
     */
    cv::Mat color(size_t view) {
        return get_view(view).color();
    }

    /**
     * Returns the depth frame image of a view.
     *
     * @param view View index.
     * @return The frame image.
     */
    cv::Mat depth(size_t view) {
        return get_view(view).depth();
    }

    /**
     * Returns the binary mask image corresponding to the region
     * occuppied by the object in a view.
     *
     * @param view  View index
     *
     * @return The binary mask image.
     */
    cv::Mat mask(size_t view) {
        return m_masks.at(view);
    }


    /**
     * Returns the tracking window of a view.
     *
     * @param i The view index.
     * @return  The tracking window rectangle.
     */
    cv::Rect track_window(int i) const {
        return trackers[i].window();
    }

    /**
     * Sets the tracking window of a view.
     *
     * This should be set prior to performing object detection in the
     * view.
     *
     * @param i         Index of the view.
     * @param bounds    The new tracking window rectangle.
     */
    void track_window(int i, const cv::Rect &bounds) {
        trackers[i].window(bounds);
    }
};


#endif //FYP_TRACKER_H
