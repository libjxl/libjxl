// Copyright (c) the JPEG XL Project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef TOOLS_COMPARISON_VIEWER_IMAGE_LOADING_H_
#define TOOLS_COMPARISON_VIEWER_IMAGE_LOADING_H_

#include <QByteArray>
#include <QImage>
#include <QString>

namespace jxl {

// `extension` should not include the dot.
bool canLoadImageWithExtension(QString extension);

// Converts the loaded image to the given display profile, or sRGB if not
// specified. Thread-hostile.
QImage loadImage(const QString& filename,
                 const QByteArray& targetIccProfile = QByteArray(),
                 const QString& sourceColorSpaceHint = QString());

}  // namespace jxl

#endif  // TOOLS_COMPARISON_VIEWER_IMAGE_LOADING_H_
