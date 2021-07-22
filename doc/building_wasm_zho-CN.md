# 构建WASM artifact

本文件主要内容为JPEG XL [Web Assembly](https://webassembly.org/)的bundle和wrapper。

以下说明假定你在使用最新的Debian、Ubuntu系统。对其他平台，或者你遇到了任何问题，请转而使用[Docker容器](developing_in_docker_zho-CN.md)。

处于追求简洁的目的，请设置以下环境变量：

 * `OPT` -包含额外软件的目录；
   Emscripten SDK的`emsdk`目录应位于此处；
   在上文提到的Docker容器中，这一变量应为`/opt`

## 必备条件

我们使用[CMake](https://cmake.org/)进行构建。若想安装CMake，请查阅[Debian构建指南](building_in_debian.md)。

为构建WebAssembly artifacts，[Emscripten SDK](https://emscripten.org/)是必须的。若想安装Emscripten SDK，请查阅[下载与安装](https://emscripten.org/docs/getting_started/downloads.html)指南：

```bash
cd $OPT

# 获取emsdk仓库
git clone https://github.com/emscripten-core/emsdk.git

# 进入目录
cd emsdk

# 下载安装SDK
./emsdk install latest

# 给当前用户激活“最新”SDK。 (写入 ~/.emscripten 文件)
./emsdk activate latest
```

[v8](https://v8.dev/)是一个用于进行测试的JavaScript引擎。和NodeJS 14相比，v8对WASM SIMD的支持更好。若想安装v8，请参考[JSVU](https://github.com/GoogleChromeLabs/jsvu)：

```bash
# 指定工作状态良好的v8版本
export v8_version="8.5.133"

# 安装JSVU
npm install jsvu -g

# 把JSVU安装到指定地点，而不是用户的“home”。
# 注意: "os" 标记应该匹配主机OS。
HOME=$OPT jsvu --os=linux64 "v8@${v8_version}"

# 将v8的二进制文件连接到无关版本的路径
ln -s "$OPT/.jsvu/v8-${v8_version}" "$OPT/.jsvu/v8"
```

在[Docker 容器](developing_in_docker_zho-CN.md)里，我们已经安装了CMake，Emscripten SDK以及V8。 

## 构建和测试项目

```bash
# 设置EMSDK和其他环境变量在实操中，EMSDK应该被设置为
# $OPT/emsdk.
source $OPT/emsdk/emsdk_env.sh

# 指定JS引擎的二进制文件
export V8=$OPT/.jsvu/v8

#使用一般WASM进行构建：
BUILD_TARGET=wasm32 emconfigure ./ci.sh release
# 或者使用SIMD WASM：
BUILD_TARGET=wasm32 ENABLE_WASM_SIMD=1 emconfigure ./ci.sh release
```
