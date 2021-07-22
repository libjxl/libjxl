# 构建和测试

本文件主要内容为`ci.sh`脚本提供的构建和测试配套工具。以下内容假定你已经设置了好了构建环境，推荐Docker（详见[指南](building_in_docker_zho-CN.md)）。

## 基础构建

构建JPEG XL软件并进行单位测试，输入以下命令：

```bash
./ci.sh release
```

## 测试

构建命令`./ci.sh`包含了`release`、`opt`等选项，并会运行测试。你可以通过设置环境变量`SKIP_TEST=1`来跳过测试。

若想手动在所有CPU上并行地进行测试，请运行如下命令：

```bash
./ci.sh test
```

为进行更快速的迭代，你也可以直接运行一个专门用于测试的二进制文件。进行测试需要通过`ctest`命令，传递给`ci.sh test`的参数会和恰当的环境变量设置一并转发给`ctest`。比如，若想列出全部可用的测试，请运行：

```bash
./ci.sh test -N
```

若想进行某一指定的测试，或者运行一系列匹配某正则表达式的测试集，请使用`ctest`的`-R`参数：

```bash
./ci.sh test -R ^MyPrefixTe
```

这一命令会运行所有以`MyPrefixTe`开头的测试。若想查看更多选项，请运行`ctest --help`命令，比如，你可以传递`-j1`参数来以单线程形式进行测试，而非默认的多测试并行模式。

## 其他命令

不带任何参数运行`./ci.sh`命令会现实可用命令的列表。比如，你可以运行`opt`来获得带有调试符号的构建，或可使用`debug`来获得未定义NDEBUG，因此包含更多运行时调试信息的调试专用构建。

### 跨平台编译

若想为有别于当前运行架构的另一架构编译代码，如果你拥有目标架构的[工具链文件](https://cmake.org/cmake/help/latest/manual/cmake-toolchains.7.html)，你可以将其传递给cmake，或可在`./ci.sh`中使用`BUILD_TARGET`环境变量。对于包括Windows 在内的一些目标，`ci.sh`会因测试需要，设置额外的环境变量。

这一操作假定你已经设置好了跨平台编译的环境，并已经为目标架构安装了的库依赖。注意，这一过程在某些情况下很难实现。因此，我们提供了一个[jpegxl docker容器](developing_in_docker_zho-CN.md)，该容器已经为跨平台编译进行了配置，也被应用在我们持续的整合工序管线之中。

比如，若想为目标三元组`aarch64-linux-gnu`编译代码，你可以运行如下命令：

```bash
BUILD_TARGET=aarch64-linux-gnu ./ci.sh release
```

当你使用`BUILD_TARGET`乃至自定义的`BUILD_DIR`这些环境变量时，必须为对 `ci.sh` 的**每一次调用**设置好这些变量，对`ci.sh test`的调用也是如此，为此，我们推荐你在shell会话中使用export设置他们，比如：

```bash
export BUILD_TARGET=x86_64-w64-mingw32 BUILD_DIR=build-foobar
```

### 格式检查（lint）

```bash
./ci.sh lint
```

Linter检查会检测你的补丁的格式是否符合项目风格。对此，我们会对被你的提交改变的代码行运行clang-format。

如果你的本地git分支正在使用`origin/master` ，且你已为你的分支进行了一些提交，运行lint命令会检查所有在共同的原始代码`origin/master`和最新变动之间的所有改变，包括未提交的改变。程序会输出应该应用到你的提交上的补丁。若想应用这些改变，请在git checkout的基础目录下运行如下命令：

```bash
./ci.sh lint | patch -p1
```

### 编程错误 (tidy)

```bash
./ci.sh tidy
```

clang-tidy是用于检查C++中常见的编程错误的工具，该工具也会检查虽然正确，但风格指南不推荐的C++结构，以及危险且可能含有漏洞的代码。

若想对你改变的文件运行clang-tidy ，你可以运行`./ci.shtidy`命令。注意，这一命令会报告被你的提交改变的所有文件里遇到的问题，而不仅仅是你提交修改的代码行。


### Address Sanitizer (asan)

```bash
./ci.sh asan
```

带有ASan的构建可以检查错误的地址用法，比如使用已释放地址的错误（use-after-free）。若想进行这些检查，以及其他未定义的表现检查，我们只需要使用上述的命令，在开启ASan之后构建代码并进行单位测试即可，非常简单。如果你想将带有ASan的构建 
常规的`build/`目录里分开，以便在asan和常规构建间快速切换，你可以以如下方式传递构建目录：

```bash
BUILD_DIR=build-asan ./ci.sh asan
```

### Memory Sanitizer (msan)

MSan可以检查运行时的非法地址访问，比如极有可能是漏洞的，对未初始化数值的访问。若想进行这些测试，你需要编译一个特别版本的项目和测试。

若要进行带有MSan的构建，你需要以`-fsanitize=memory`参数构建libc++，以供我们在带有MSan的构建之中对其进行连接。此外，安装`llvm-symbolizer`也是很有帮助的，其可帮助获取带有调试符号（函数和行编号）的堆栈追踪信息。若想在Debian系系统上安装`llvm-symbolizer`，请运行如下命令：

```bash
sudo apt install llvm # 或者llvm-7等，选择指定版本。
```

若想安装以`-fsanitize=memory`参数编译的libc++，你可以使用`./ci.sh msan_install`捷径命令。这一命令会下载，编译并在`${HOME}/.msan`目录安装libc++以及libc++abi，以供日后使用。

在设置好这些之后，你可以通过如下命令构建项目：

```bash
./ci.sh msan
```

这一命令默认会使用`build`目录来储存cmake和对象文件。如果你想要为带有msan的构建使用一个不同的构建目录你可以运行如下命令：

```bash
BUILD_DIR=build-msan ./ci.sh msan
```
