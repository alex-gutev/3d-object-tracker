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

#ifndef FYP_APP_UI_H
#define FYP_APP_UI_H

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "tracker.h"

/**
 * Handles the general application logic and user interface of the tracker
 * application.
 */
class tracker_app {

    /**
     * OpenCV window name identifier prefix
     */
    static constexpr const char * const window_name = "tracker_view";

    /**
     * True if the region of interest has been selected.
     */
    bool has_roi = false;

    /**
     * True if the application is currently tracking the object.
     */
    bool tracking = false;

    /**
     * Array of OpenCV view window identifiers.
     *
     * There is one window per view. The view's window is located at
     * the element at the view's index with the array.
     */
    std::vector<std::string> windows;

    /**
     * Array of the images displayed on each view's window.
     *
     * The view's image is stored at the element at the view's index
     * within the array.
     */
    std::vector<cv::Mat> images;

    /**
     * Tracker object.
     *
     * Responsible for object detection and tracking.
     */
    tracker track;


    /** Methods */

    /**
     * Parse the view parameters from the file at @a path.  Creates a
     * view object, window and image for each view parsed. There is no
     * upper limit on the number of views.
     *
     * The following information is parsed from the file:
     *
     * - Path to the colour and depth video files.
     * - Video dimensions.
     * - Z near and Z far values.
     * - Fundamental and intrinsic matrices.
     *
     * @param path The path to the view info file
     */
    void parse_views(const std::string &path);

    /**
     * Creates a new view window and image. Appends the window
     * identifier and empty image matrix to the @a windows and @a
     * images arrays respectively.
     */
    void create_view_window();

    /**
     * Reads a frame from the colour and depth videos of each view and
     * copies the colour images to the corresponding view image in the
     * @a images array.
     *
     * @return True if a colour and depth image was read from each
     *         view's video file, false if the end of file was
     *         reached.
     */
    bool read_images();

    /**
     * Copies the current frame image of each view's colour video to
     * the corresponding image within the @a images array.
     */
    void clone_images();

    /**
     * Display's each view's image (each image in the @a images array)
     * on the view's window.
     */
    void show_images();


    /* Object detection */

    /**
     * Detects objects within the selected region of interest in the
     * primary view.  The object region is mapped to the remaining
     * views and overlayed onto each view's image.
     *
     * @param primary_view The index of the view in which to detect ojects.
     */
    void detect_objects(size_t primary_view, bool auto_threshold);

    /**
     * Displays the object region in each view, as a transparent
     * overlay onto the frame image.
     */
    void display_regions();


    /* Object Tracking */

    /**
     * Loads the next video frames and tracks the object in each view.
     * The updated position of the tracking window is displayed.
     *
     * @return True if the next frame was read successfully, false if
     *         the end of file was reached.
     */
    bool track_next();

    /**
     * Displays each view's tracking window onto each view's frame
     * image.
     */
    void display_track_windows();


    /* Event processing */

    /**
     * Mouse event callback function.
     *
     * @param ctx Index of the window on which this event occurred.
     */
    static void mouse_callback(int event, int x, int y, int flags, void *ctx);

    /**
     * Mouse event handler method.
     */
    void on_mouse_event(int event, int x, int y, int flags, size_t view_index);

    /**
     * Processes keyboard events. Returns when the user terminates the
     * application.
     */
    void wait_key();

    /**
     * Performs object tracking throughout the entire video
     * sequence. Returns when the object has been tracked in the last
     * frame of the video.
     */
    void tracking_loop();

    /**
     * Constructor. Private to prevent the object from being
     * instantiated as it is meant to be used as a singleton.
     */
    tracker_app() = default;

    /**
     * Parses the remaining command-line arguments.
     *
     * @param argc Number of arguments
     * @param argv Array of arguments
     */
    void parse_args(int argc, char **argv);

public:

    /**
     * @return The singleton instance.
     */
    static tracker_app *instance();

    /**
     * Runs the tracker application and begins processing
     * events. Returns once the user terminates the application
     *
     * @param argc  Number of command-line arguments
     * @param argv  Array of command-line arguments.
     */
    void run(int argc, char **argv);
};


#endif //FYP_APP_UI_H
