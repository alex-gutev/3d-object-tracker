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

#ifndef VIEW_TRACKER_H
#define VIEW_TRACKER_H

#include <tuple>

#include <opencv2/imgproc.hpp>

#include "view.h"

/**
 * Single view Object tracker.
 */
class view_tracker {
public:
    /* Accessors */

    /**
     * Returns the underlying 'view' struct.
     */
    view &view_info() {
        return m_view;
    }

    /**
     * Returns the tracking window bounds.
     *
     * @return cv::Rect - Rectangle
     */
    cv::Rect window() const {
        return m_window;
    }
    /**
     * Sets the tracking window bounds.
     *
     * @param window The tracking window rectangle.
     */
    void window(cv::Rect window) {
        m_window = window;
    }

    /**
     * Returns the position of the window in the z dimension.
     */
    float window_z() const {
        return m_window_z;
    }
    /**
     * Sets the position of the window in the z dimension.
     *
     * @param z The z position
     */
    void window_z(float z) {
        m_window_z = z;
    }


    /* Building the Model */

    /**
     * Builds the appearance model.
     *
     * @param mask The mask indicating which pixels belong to the
     *   object.
     */
    void build_model(cv::Mat mask);

    /**
     * Estimates the mean-shift bandwidth.
     */
    void estimate_bandwidth();

    /**
     * Set the mean-shift bandwidth.
     *
     * @param h The bandwidth.
     */
    void bandwidth(float h);

    /* Tracking */

    /**
     * Track the object in the current frame.
     *
     * @return A weight indicating the estimated accuracy of the
     *   tracked position.
     */
    float track();

private:
    /**
     * Stores the view parameters and reads the colour and depth frame
     * images.
     */
    view m_view;

    /**
     * Colour Histogram.
     */
    cv::Mat hist;

    /**
     * Mean-shift bandwidth.
     */
    double h;

    /**
     * Range of the object in the z-dimension.
     */
    double z_range;


    /**
     * Tracking window rectangle.
     */
    cv::Rect m_window;

    /**
     * Position of the tracking window in the z dimension.
     */
    float m_window_z;


    /**
     * Backprojects the colour histogram onto the colour frame image.
     *
     * @return A single-channel probability image in which each
     *         pixel's value is the probability that it belongs to the
     *         object.
     */
    cv::Mat backproject();

    float compute_area_covered();

    /**
     * Performs 3D Mean Shift in the point-cloud space of view @a v's
     * colour frame image projected into 3D space using the disparity
     * image.
     *
     * Mean shift is performed in the point-cloud to find the centre
     * of the closest region (based on the 3D Euclidean Distance) with
     * pixel colours which have a high probability of belonging to the
     * object based on the object's colour histogram. The pixel
     * probabilities are provided by @a prob_img.
     *
     * A Gaussian kernel, with bandwidth parameter @a h, is used.
     *
     * @param prob_img      The probability image, a single channel image where the value
     *                      of each pixel is the probability that the pixel belongs to the
     *                      object.
     *
     * @param v             The view in which tracking is to be performed.
     *
     * @param window        The tracking window (the centre will be used as the starting position).
     *
     * @param depth         The depth of the centre of the tracking window.
     *
     * @param num_iters     Number of iterations to perform before stopping.
     *
     * @param epsilon       The minimum shift (between the new and old positions) to be considered significant.
     *                      If the shift is below this values the loop is terminated before num_iters iterations
     *                      are performed. The shift is in the unit of the 3D Euclidean distance.
     *
     * @param h             Gaussian kernel bandwidth.
     *
     * @return A pair where the first value is the new tracking window
     *         (the centre of which is the object's new position) and
     *         the second value is the depth of the object.
     */
    static std::pair<cv::Rect, float> mean_shift(cv::Mat prob_img,
                                                 view &v,
                                                 cv::Rect window,
                                                 float depth,
                                                 int num_iters,
                                                 float epsilon,
                                                 float h);
};


#endif /* VIEW_TRACKER_H */

// Local Variables:
// mode: c++
// End:
