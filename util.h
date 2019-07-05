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

#ifndef AGTRACK_UTIL_H
#define AGTRACK_UTIL_H

/**
 * Utility functions used throughout the project.
 */

#include <algorithm>

/**
 * Clamps a value to the range [@a min, @a max]
 *
 * @param x The value to clamp.
 * @param min The lower-bound of the range.
 * @param max The upper-bound of the range.
 *
 * @return The clamped value
 */
template<typename T>
T clamp(T x, T min, T max) {
    return std::min(std::max(x, min), max);
}

#endif /* AGTRACK_UTIL_H */

// Local Variables:
// mode: c++
// End:
