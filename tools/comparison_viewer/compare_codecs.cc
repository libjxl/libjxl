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

#include <stdlib.h>

#include <QApplication>
#include <QMessageBox>
#include <QString>
#include <QStringList>

#include "tools/comparison_viewer/codec_comparison_window.h"

namespace {

template <typename Window, typename... Arguments>
void displayNewWindow(Arguments&&... arguments) {
  Window* const window = new Window(arguments...);
  window->setAttribute(Qt::WA_DeleteOnClose);
  window->show();
}

}  // namespace

int main(int argc, char** argv) {
  QApplication application(argc, argv);

  QStringList arguments = application.arguments();
  arguments.removeFirst();  // program name

  if (arguments.empty()) {
    QMessageBox message;
    message.setIcon(QMessageBox::Information);
    message.setWindowTitle(
        QCoreApplication::translate("CodecComparisonWindow", "Usage"));
    message.setText(QCoreApplication::translate(
        "CodecComparisonWindow", "Please specify a directory to use."));
    message.setDetailedText(QCoreApplication::translate(
        "CodecComparisonWindow",
        "That directory should contain images in the following layout:\n"
        "- .../<image name>/original.png (optional)\n"
        "- .../<image_name>/<codec_name>/<compression_level>.<ext>\n"
        "- .../<image_name>/<codec_name>/<compression_level>.png (optional for "
        "formats that Qt can load)\n"
        "With arbitrary nesting allowed before that. (The \"...\" part is "
        "referred to as an \"image set\" by the tool."));
    message.exec();
    return EXIT_FAILURE;
  }

  for (const QString& argument : arguments) {
    displayNewWindow<jxl::CodecComparisonWindow>(argument);
  }

  return application.exec();
}
