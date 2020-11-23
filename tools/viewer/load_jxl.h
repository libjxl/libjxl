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

#ifndef TOOLS_VIEWER_LOAD_JXL_H_
#define TOOLS_VIEWER_LOAD_JXL_H_

#include <QByteArray>
#include <QImage>
#include <QString>

namespace jxl {

QImage loadJxlImage(const QString& filename, const QByteArray& targetIccProfile,
                    qint64* elapsed, bool* usedRequestedProfile = nullptr);

}  // namespace jxl

#endif  // TOOLS_VIEWER_LOAD_JXL_H_
