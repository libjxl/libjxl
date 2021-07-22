# 使用64-bit Windows开发

以下说明假定你在使用最新的Windows 10系统（比如build 19041.928），且已安装Microsoft Visual Studio 2019（比如Version 16.9.0 Preview 4.0）。如果以上要求不可用，请转而使用[Docker容器](developing_in_docker_zho-CN.md)。

## 最小构建依赖

除了third_party里面的依赖，有些工具使用了其他的外部依赖，你首先需要在你的系统上安装它们。

请[安装vcpkg](https://vcpkg.readthedocs.io/en/latest/examples/installing-and-using-packages/)(version 2019.07.18测试通过)，并使用该工具安装以下库：

```
vcpkg install gtest:x64-windows
vcpkg install giflib:x64-windows
vcpkg install libjpeg-turbo:x64-windows
vcpkg install libpng:x64-windows
vcpkg install zlib:x64-windows
```

## 构建

在Visual Studio中,打开JPEG XL根目录中的CMakeLists.txt。在解决方案浏览器中的文件夹视图里右键点击CMakeLists.txt。在右键菜单中选择CMake 设置。点击绿色加号来添加x64-Clang配置，并使用红色减号来移除任何非Clang的配置（暂不支持MSVC编译器）。在点击蓝色的超链接“CMakeSettings.json”之后，一个编辑页面会出现。插入以下内容，并将$VCPKG替换为你安装上文vcpkg的地址。

```
{
  "configurations": [
    {
      "name": "x64-Clang-Release",
      "generator":"Ninja",
      "configurationType":"MinSizeRel",
      "buildRoot": "${projectDir}\\out\\build\\${name}",
      "installRoot": "${projectDir}\\out\\install\\${name}",
      "cmakeCommandArgs": "-DCMAKE_TOOLCHAIN_FILE=$VCPKG/scripts/buildsystems/vcpkg.cmake",
      "buildCommandArgs": "-v",
      "ctestCommandArgs": "",
      "inheritEnvironments": [ "clang_cl_x64" ],
      "variables": [
        {
          "name":"VCPKG_TARGET_TRIPLET",
          "value": "x64-windows",
          "type":"STRING"
        },
        {
          "name":"JPEGXL_ENABLE_TCMALLOC",
          "value":"False",
          "type":"BOOL"
        },
        {
          "name":"BUILD_GMOCK",
          "value":"True",
          "type":"BOOL"
        },
        {
          "name": "gtest_force_shared_crt",
          "value":"True",
          "type":"BOOL"
        },
        {
          "name":"JPEGXL_ENABLE_FUZZERS",
          "value":"False",
          "type":"BOOL"
        },
        {
          "name":"JPEGXL_ENABLE_VIEWERS",
          "value":"False",
          "type":"BOOL"
        }
      ]
    }
  ]
}
```

本项目现已准备完毕。可按F7进行构建（或者选择构建菜单里面的构建全部）。这一命令会将二进制文件写入`out/build/x64-Clang-Release/tools`。主[README文档](README.md)解释了使用编、解码器和基准测试二进制文件的方法。
