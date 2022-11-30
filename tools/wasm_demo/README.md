## WebAssembly demonstration

This folder contains an example how to decode JPEG XL files on a web page using
WASM engine.

### Hosting

To enable multi-threading some files should be served in a secure context (i.e.
transferred over HTTPS) and executed in a "site-isolation" mode (controlled by
COOP and COEP response headers).

Unfortunately [GitHub Pages](https://pages.github.com/) does not allow setting
response headers.

[Netlify](https://www.netlify.com/) provides free, easy to setup and deploy
platform for serving such demonstration sites. However, any other
service provider / software that allows changing response headers could be
employed as well.
