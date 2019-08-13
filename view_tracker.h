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
#include <opencv2/ml.hpp>

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
     * @param predicted The predicted position.
     *
     * @return A weight indicating the estimated accuracy of the
     *   tracked position.
     */
    float track(cv::Point3f predicted);

private:
    /* Model */

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


    /* Position State */

    /**
     * Tracking window rectangle.
     */
    cv::Rect m_window;

    /**
     * Position of the tracking window in the z dimension.
     */
    float m_window_z;


    /* Detected Objects */

    /**
     * Stores information about object detected within the tracking
     * window.
     */
    struct object {
        enum object_type {
            /* Unknown Object Type */
            type_unknown = 0,

            /* Object forms part of background */
            type_background = 1,
            /* Occluding Object */
            type_occluder,

            /* Part of the target object */
            type_target,
        };

        /**
         * Constant indicating the object type.
         */
        object_type type;

        /**
         * Depth of the pixel closest to the camera.
         */
        float min;
        /**
         * Depth of the pixel furthest from the camera.
         */
        float max;

        /**
         * Median depth of the pixels. Indicates position of object in
         * Z axis.
         */
        float depth;

        /**
         * Position of object's centre in 3D world space.
         */
        cv::Point3f pos;

        /**
         * Mask identifying pixels belonging to the object.
         */
        cv::Mat region;

        /**
         * Bounding rectangle of the object.
         */
        cv::Rect bounds;

        /**
         * Constructor.
         *
         * @param type Object type.
         * @param min Depth of pixel closest to camera.
         * @param max Depth of pixel furthest from camera.
         * @param depth Median pixel depth.
         */
        object(object_type type, float min, float max, float depth) :
            type(type), min(min), max(max), depth(depth) {}
    };

    /**
     * List of objects detected, within the tracking window, in the
     * previous frame.
     */
    std::vector<object> objects;


    /* Occlusion Detection */

    /**
     * Determines whether the object is occluded in the current frame.
     *
     * @param window The tracking window
     *
     * @param z The new z position of the object as determined by the
     *    mean shift tracker.
     *
     * @param predicted The predicted position of the object.
     *
     * @return True if the object is occluded, false otherwise.
     */
    bool is_occluded(cv::Rect window, float z, cv::Point3f predicted);

    /**
     * Segments the current frame image, within the region @a r, to
     * detect the objects currently in the scene.
     *
     * @param r The region to segment.
     *
     * @return List of detected objects.
     */
    std::vector<object> detect_objects(cv::Rect r) const;

    /**
     * Matches the objects detected in the previous frame, in the
     * 'objects' array, to the objects detected in the current frame,
     * in the array @a new_objects.
     *
     * For each object, in @a new_objects, which is matched to an
     * object in the previous frame, its type is set to that of the
     * object in the previous frame.
     *
     * @param new_objects Array of objects to be matched.
     * @param r Region in which the new objects were detected.
     */
    void match_objects(std::vector<object> &new_objects, cv::Rect r) const;


    /* Mean Shift Tracking */

    /**
     * Backprojects the colour histogram onto the colour frame image.
     *
     * @return A single-channel probability image in which each
     *         pixel's value is the probability that it belongs to the
     *         object.
     */
    cv::Mat backproject();

    /**
     * Computes a rough estimate of the area covered by the object,
     * within the tracking window.
     *
     * @return The area covered in pixels.
     */
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
