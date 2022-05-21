# JPEG XL software support

This document attempts to keep track of software that is using libjxl to support JPEG XL.
This list serves several purposes:

- thank/acknowledge other projects for integrating jxl support
- point end-users to software that can read/write jxl
- keep track of the adoption status of jxl
- in case of a (security) bug in libjxl, it's easier to see who might be affected and check if they are updated (in case they use static linking)

Please add missing software to this list.

## Browsers

- Chromium: behind a flag since version 91, [tracking bug](https://bugs.chromium.org/p/chromium/issues/detail?id=1178058)
- Firefox: behind a flag since version 90, [tracking bug](https://bugzilla.mozilla.org/show_bug.cgi?id=1539075)
- Safari: not supported, [tracking bug](https://bugs.webkit.org/show_bug.cgi?id=208235)
- Edge: behind a flag since version 91, start with `.\msedge.exe --enable-features=JXL`
- Opera: behind a flag since version 77.
- For all browsers and to track browsers progress see [Can I Use](https://caniuse.com/jpegxl).

## Image libraries

- [ImageMagick](https://imagemagick.org/): supported since 7.0.10-54
- [libvips](https://libvips.github.io/libvips/): supported since 8.11
- [Imlib2](https://github.com/alistair7/imlib2-jxl)
- [FFmpeg](https://github.com/FFmpeg/FFmpeg/search?q=jpeg-xl&type=commits)
- [GDAL](https://gdal.org/drivers/raster/jpegxl.html): supported since 3.4.0 as a TIFF codec, and 3.6.0 as standalone format

## OS-level support / UI frameworks / file browser plugins

- Qt / KDE: [plugin available](https://github.com/novomesk/qt-jpegxl-image-plugin)
- GDK-pixbuf: plugin available in libjxl repo
- [gThumb](https://ubuntuhandbook.org/index.php/2021/04/gthumb-3-11-3-adds-jpeg-xl-support/)
- [MacOS viewer/QuickLook plugin](https://github.com/yllan/JXLook)
- [Windows Imaging Component](https://github.com/mirillis/jpegxl-wic)
- [Windows thumbnail handler](https://github.com/saschanaz/jxl-winthumb)
- [OpenMandriva Lx (since 4.3 RC)](https://www.openmandriva.org/en/news/article/openmandriva-lx-4-3-rc-available-for-testing)
- [KaOS (since 2021.06)](https://news.itsfoss.com/kaos-2021-06-release/)
- [EFL (since 1.27, no external plugin needed)](https://www.enlightenment.org)

## Image editors

- [GIMP (since 2.99.8)](https://www.gimp.org/news/2021/10/20/gimp-2-99-8-released/); plugin for older versions available in libjxl repo
- [Krita](https://invent.kde.org/graphics/krita/-/commit/13e5d2e5b9f0eac5c8064b7767f0b62264a0797b)
- Photoshop: no plugin available yet, no official support yet

## Image viewers

- [XnView](https://www.xnview.com/en/)
- [ImageGlass](https://imageglass.org/)
- [IrfanView](https://www.irfanview.com/); supported since 4.59 - requires a [plugin](https://www.irfanview.com/plugins.htm) to be downloaded and enabled.
- [Tachiyomi](https://github.com/tachiyomiorg/tachiyomi/releases/tag/v0.12.1)
- Any viewer based on Qt, KDE, GDK-pixbuf, EFL, ImageMagick, libvips or imlib2 (see above)
  - Qt viewers: gwenview, digiKam, KolourPaint, KPhotoAlbum, LXImage-Qt, qimgv, qView, nomacs, VookiImageViewer, PhotoQt
  - GTK viewers: Eye of Gnome (eog), gThumb, Geeqie
  - EFL viewers: entice, ephoto
- [Swayimg](https://github.com/artemsen/swayimg)

## Online tools

- [Squoosh](https://squoosh.app/)
- [Cloudinary](https://cloudinary.com/blog/cloudinary_supports_jpeg_xl)
- [MConverter](https://mconverter.eu/)
- [jpegxl.io](https://jpegxl.io/)
