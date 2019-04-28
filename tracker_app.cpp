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

#include "tracker_app.h"

#include <iostream>

#include "view_parser.h"


void tracker_app::mouse_callback(int event, int x, int y, int flags, void *ctx) {
    tracker_app::instance()->on_mouse_event(event, x, y, flags, (size_t)ctx);
}

void tracker_app::on_mouse_event(int event, int x, int y, int flags, size_t view_index) {
    if (event == cv::EVENT_LBUTTONUP) {
        if (!has_roi) {
            // Select ROI in chosen view
            track.track_window(view_index, cv::selectROI(windows[view_index], images[view_index], false, false));
            has_roi = true;

            detect_objects(view_index, true);

            // Reset mouse callback as cv::selectROI removes it
            cv::setMouseCallback(windows[view_index], tracker_app::mouse_callback, (void*)view_index);
        }
    }
}

void tracker_app::detect_objects(size_t primary_view, bool auto_threshold) {
    track.detect_objects(primary_view);

    track.build_models();
    track.primary_view(primary_view);

    tracking = true;

    display_regions();
}

void tracker_app::display_regions() {
    size_t i = 0;
    for (cv::Mat &img : images) {
        cv::Mat red(img.size(), img.type());
        cv::Mat region;

        red = cv::Scalar(0,0,255);
        red.copyTo(region, track.mask(i));

        cv::addWeighted(track.color(i), 0.5, region, 0.5, 0.0, img);

        cv::imshow(windows[i], img);
        i++;
    }
}


/// Tracking

bool tracker_app::track_next() {
    if (read_images()) {
        show_images();
        track.track();

        display_track_windows();

        return true;
    }

    return false;
}

void tracker_app::display_track_windows() {
    for (int i = 0; i < windows.size(); i++) {
        cv::rectangle(images[i], track.track_window(i), cv::Scalar(0,0,255), 5);
        cv::imshow(windows[i], images[i]);
    }
}


/// Parsing view information

void tracker_app::parse_views(const std::string &path) {
    view_parser parser(path);

    while (parser.next()) {
        parser.parse(track.add_view());
        create_view_window();
    }
}

void tracker_app::create_view_window() {
    size_t index = windows.size();
    std::string name(window_name);

    // Create unique window identifier string
    name.append(std::to_string(index));

    windows.push_back(name);
    images.emplace_back();

    cv::namedWindow(name, cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO);
    cv::setMouseCallback(name, mouse_callback, (void *)index);
}


/// Reading images

bool tracker_app::read_images() {
    if (track.read()) {
        clone_images();
        return true;
    }

    return false;
}

void tracker_app::show_images() {
    size_t i = 0;

    for (std::string &name : windows) {
        cv::imshow(name, images[i++]);
    }
}

void tracker_app::clone_images() {
    for (size_t i = 0, num_views = windows.size(); i < num_views; i++) {
        images[i] = track.color(i).clone();
    }
}


/// Event processing

void tracker_app::wait_key() {
    int key;

    // Until escape key is pressed
    while ((key = cv::waitKey(0)) != 27) {
        switch (key) {
            case 13:
                if (tracking) {
                    tracking_loop();
                }
                break;
        }
    }
}

void tracker_app::tracking_loop() {
    track.estimate_bandwidth();

    while (track_next()) {
        // Pass control to OpenCV HighGUI to allow the window to be
        // repainted, this requires a small delay in the waitKey
        // function, 5 milliseconds is chosen.
        cv::waitKey(5);
    }

    // Print execution time statistics
    track.print_time_stats();
}


/// Public Methods

tracker_app *tracker_app::instance() {
    static tracker_app *inst = new tracker_app();
    return inst;
}

void tracker_app::run(int argc, char **argv) {
    if (argc < 2) {
        std::cerr << "Usage tracker [params file]\n";
        return;
    }

    // Parse views from view data file passed as first argument
    parse_views(argv[1]);

    if (read_images()) {
        show_images();

        // Parse remaining commandline arguments
        parse_args(argc - 2, argv + 2);

        // Begin application event loop
        wait_key();
    }
    else {
        std::cerr << "Error: No frames in videos\n";
    }
}

void tracker_app::parse_args(int argc, char **argv) {
    if (argc >= 5) {
        // Parse primary view
        int view = atoi(argv[0]);
        // Parse X of top-left of tracking window
        int x = atoi(argv[1]);
        // Parse Y of top-left of tracking window
        int y = atoi(argv[2]);
        // Parse width of tracking window
        int width = atoi(argv[3]);
        // Parse height of tracking window
        int height = atoi(argv[4]);

        // Set tracking window
        track.track_window(view, cv::Rect(x, y, width, height));
        has_roi = true;

        // Perform object detection in the tracking window
        detect_objects(view, true);
    }
}
