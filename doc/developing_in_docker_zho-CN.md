# 使用docker开发

Docker 允许在一个打包的容器内运行软件，和宿主系统隔离。这一性质可让我们在标准环境中运行代码，而不必处理开发中引入的不同构建环境带来的问题，同时也以在标准环境中自动设置的方式简化了外部依赖的安装。

## 设置容器

请阅读[docker安装指导](https://docs.docker.com/install/) 来在你的电脑上安装docker。截止至2020-01，这需要你创建一个免费的Docker账户。

我们的构建器使用的镜像是一份包含了所需依赖和构建工具的ubuntu-bionic镜像。你可以从`gcr.io/jpegxl/jpegxl-builder` 拉取镜像，具体命令如下：

```bash
sudo docker pull gcr.io/jpegxl/jpegxl-builder
```

若想使用该Docker镜像，请运行以下命令：

```bash
sudo docker run -it --rm \
  --user $(id -u):$(id -g) \
  -v $HOME/jpeg-xl:/jpeg-xl -w /jpeg-xl \
  gcr.io/jpegxl/jpegxl-builder bash
```

这会创建并启动一个镜像，注意，该镜像在你退出终端之后会被自动删除（`--rm` 参数）。

`-v`用于将包含jpeg-xl代码的主机目录 （假设为`$HOME/jpeg-xl`）映射到容器内部的/jpeg-xl 目录。这意味着，当你用自己喜爱的编辑器编辑主机代码的同时，这些变化也会被传递到容器当中，因为这一目录只是被单纯的挂载到了容器里。

对OSX，这一路径必须是对Docker白名单的、共享的路径之一。$HOME（/Users/ 的子目录）是在Docker出厂设置下正常工作的目录之一。

在OSX上，“cannot find name for group ID”这一错误可以被忽视。

在Windows系统中，你可以在从Gitlab获取的jpeg-xl目录中直接运行以下命令：

```bash
docker run -u root:root -it --rm -v %cd%:/jpeg-xl -w /jpeg-xl \
  gcr.io/jpegxl/jpegxl-builder
```

## 基础构建

在Docker容器内部，你可以使用以下命令编译并进行单位测试

```bash
CC=clang-7 CXX=clang++-7 ./ci.sh opt
```

这一命令会将二进制文件写入`/jpeg-xl/build/tools`目录并运行单位测试。可参考[构建模式和测试](building_and_testing_zho-CN.md)来获取更多信息。

如果主机上已经有了一个build目录，这可能会导致冲突，使用`rm -rf build`来删除它。

注意默认的“clang”编译器没有被装到镜像里，因此我们指定clang-7。如果 build/ 目录已经存在，且以被另一个不同编译器使用，cmake会报错。你可以通过对已有的 build/ 目录进行重命名来避免这一错误，也可以设置`BUILD_DIR`环境变量。

## 跨平台编译环境（可选）

我们在主要的Docker镜像`jpegxl-builder`里安装了跨平台编译所需的必要工具。这提供了在qemu环境下，为包含arm在内的其他架构进行编译并运行测试的能力。

该docker容器已经有了一些 `qemu-*-static`二进制文件（比如`qemu-aarch64-static`），这些文件可用于在x86_64系统上模拟其他架构。若想实现在容器内部运行异构程序时，系统可自动使用这些qemu二进制文件，你需要安装 `binfmt`并设置系统使用*宿主*中的`/usr/bin/qemu-*-static`二进制文件。这是Ubuntu、Debian的默认目录，然而无论是否使用了宿主的qemu-user-static二进制文件，你都需要在宿主上同时安装`binfmt-support` 以及`qemu-user-static`，因为Debian、Ubuntu的binfmt-support只会配置在宿主机上已有对应qemu-user-static二进制文件的架构的`binfmt` 签名。如果你在别的发行版中，将其配置到了其他目录，请在运行前将其软连接到docker内部的`/usr/bin/qemu-*-static`目录。若想安装你Ubuntu宿主机支持的binfmt，请在容器*外*运行以下命令：

```bash
sudo apt install binfmt-support qemu-user-static
```

之后通过以下命令进行跨平台编译以及单位测试：

```bash
export BUILD_TARGET=aarch64-linux-gnu CC=clang-7 CXX=clang++-7
./ci.sh release
```

环境变量 `BUILD_TARGET=aarch64-linux-gnu`指示`ci.sh`脚本以该架构为目标进行跨平台编译为避免和宿主`build`混淆的问题，这也会将默认的`BUILD_DIR` 改为`build-aarch64` 。你也可以显式的设置一个`BUILD_DIR`环境变量，这一设置会被优先使用。本容器中受支持的BUILD_TARGET值如下：

*    *the empty string* (原生x86_64支持)
*    aarch64-linux-gnu
*    arm-linux-gnueabihf
*    i686-linux-gnu
*    x86_64-w64-mingw32 (Windows构建)

注意，对想在docker container之外测试Windows构建的情况，请阅读[使用64-bit Windows开发](developing_in_windows_zho-CN.md)指南中的 最小构建依赖 部分，并安装所需的外部依赖。
