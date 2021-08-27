# 使用Debian开发

以下说明假定你在使用最新的Debian、Ubuntu系统。其他平台请转而使用[Docker容器](developing_in_docker.sc.md)。

## 最小构建依赖

除了third_party里面的依赖，有些工具使用了其他的外部依赖，你首先需要在你的系统上安装它们。

```bash
sudo apt install cmake clang doxygen g++-8 extra-cmake-modules libgif-dev \
  libjpeg-dev ninja-build libgoogle-perftools-dev
```

确保你的默认“clang”编译器版本号大于6。

```bash
clang --version
```

在输出老版本的情况下，哪怕你已经安装了clang-7，你也需要升级默认的clang编译器。在Debian系系统中，运行以下命令：

```bash
sudo update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-7 100
sudo update-alternatives --install /usr/bin/clang clang /usr/bin/clang-7 100
```

可选的，若要编译一些额外的支持工具和测试，你可以安装以下软件包：

```bash
sudo apt install qtbase5-dev libqt5x11extras5-dev libwebp-dev libgimp2.0-dev \
  libopenexr-dev libgtest-dev libgmock-dev libbenchmark-dev libbenchmark-tools
```

若想使用lint、coverage命令，你同样需要额外软件包：

```bash
sudo apt install clang-format clang-tidy curl parallel gcovr
```

## 构建

本项目使用Cmake进行构建。我们提供了一个脚本来简化调用。若想对项目进行构建和测试，运行以下命令：

```bash
./ci.sh opt
```

这一命令会将二进制文件写入`build/tools`目录并运行单位测试。请查阅[构建模式和测试](building_and_testing.sc.md)来获取更多信息。
