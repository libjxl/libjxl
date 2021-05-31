# NGINX JPEG XL modules
This folder provides a module for serving losslessly recompressed JPEG
files in a transparent way, as well as a script `prepare_folder.sh` that
can prepare losslessly-recompressed .jxl files for all the .jpg files in
a folder; nginx can then configured to serve one file or the other depending
on the client.

This plugins is highly experimental, and can be compiled as other nginx plugins,
running from the source directory of nginx:

```
auto/configure --with-compat --add-dynamic-module=path_to_nginx_jxl/ && make modules
```
