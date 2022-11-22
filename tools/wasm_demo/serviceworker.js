// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
 * ServiceWorker script.
 *
 * Multi-threading in WASM is currently implemented by the means of
 * SharedArrayBuffer. Due to infamous vulnerabilities this feature is disabled
 * unless site is running in "cross-origin isolated" mode.
 * If there is not enough control over the server (e.g. when pages are hosted as
 * "github pages") ServiceWorker is used to upgrade responses with corresponding
 * headers.
 *
 * This script could be executed in 2 environments: HTML page or ServiceWorker.
 * The environment is detected by the type of "window" reference.
 *
 * When this script is executed from HTML page then ServiceWorker is registered.
 * Page reload might be necessary in some situations. By default it is done via
 * `window.location.reload()`. However this can be altered by setting a
 * configuration object `window.serviceWorkerConfig`. It's `doReload` property
 * should be a replacement callable.
 *
 * When this script is executed from ServiceWorker then standard lifecycle
 * event dispatchers are setup along with `fetch` interceptor.
 */

(() => {
  // Embedded (baked-in) responses for faster turn-around.
  const EMBEDDED = {
    'jxl_decoder.js': '$jxl_decoder.js$',
    'jxl_decoder.worker.js': '$jxl_decoder.worker.js$',
  };

  const setCopHeaders = (headers) => {
    headers.set('Cross-Origin-Embedder-Policy', 'require-corp');
    headers.set('Cross-Origin-Opener-Policy', 'same-origin');
  };

  const maybeProcessEmbeddedResources = (event) => {
    const url = event.request.url;
    // Shortcut for baked-in scripts.
    for (const [key, value] of Object.entries(EMBEDDED)) {
      if (url.endsWith(key)) {
        const headers = new Headers();
        headers.set('Content-Type', 'application/javascript');
        setCopHeaders(headers);

        event.respondWith(new Response(value, {
          status: 200,
          statusText: 'OK',
          headers: headers,
        }));
        return true;
      }
    }
    return false;
  };

  const upgradeResponse = (response) => {
    if (response.status === 0) {
      return response;
    }

    const newHeaders = new Headers(response.headers);
    setCopHeaders(newHeaders);

    return new Response(response.body, {
      status: response.status,
      statusText: response.statusText,
      headers: newHeaders,
    });
  };

  const onFetch = (event) => {
    // Pass direct cached resource requests.
    if (event.request.cache === 'only-if-cached' &&
        event.request.mode !== 'same-origin') {
      return;
    }

    // Serve backed resources.
    if (maybeProcessEmbeddedResources(event)) {
      return;
    }

    // Pass HTML page fetching.
    if (event.request.destination !== 'document') {
      return;
    }

    // Upgrade all other request responses.
    event.respondWith(fetch(event.request)
                          .then(upgradeResponse)
                          .catch((err) => console.error(err)));
  };

  const serviceWorkerMain = () => {
    // ServiceWorker lifecycle.
    self.addEventListener('install', () => self.skipWaiting());
    self.addEventListener(
        'activate', (event) => event.waitUntil(self.clients.claim()));
    self.addEventListener('message', (ev) => {
      if (ev.data && ev.data.type === 'deregister') {
        self.registration.unregister()
            .then(() => {
              return self.clients.matchAll();
            })
            .then(clients => {
              clients.forEach((client) => client.navigate(client.url));
            });
      }
    });
    // Intercept some requests.
    self.addEventListener('fetch', onFetch);
  };

  // Executed in HTML page environment.
  const maybeRegisterServiceWorker = () => {
    if (!window.isSecureContext) {
      config.log('Secure context is required for this ServiceWorker.');
      return;
    }

    const config = {
      doReload: () => window.location.reload(),
      log: console.log,
      error: console.error,
      ...window.serviceWorkerConfig  // add overrides
    }

    const n = navigator;
    // In some environments this won't be available.
    if (!n.serviceWorker) {
      return;
    }

    const onServiceWorkerRegistrationSuccess = (registration) => {
      config.log('Service Worker registered', registration.scope);

      registration.addEventListener('updatefound', () => {
        config.log('Reloading page to make use of updated Service Worker.');
        config.doReload();
      });

      // If the registration is active, but it's not controlling the
      // page
      if (registration.active && !n.serviceWorker.controller) {
        config.log('Reloading page to make use of Service Worker.');
        config.doReload();
      }
    };

    const onServiceWorkerRegistrationFailure = (err) => {
      config.error('Service Worker failed to register:', err);
    };

    n.serviceWorker.register(window.document.currentScript.src)
        .then(
            onServiceWorkerRegistrationSuccess,
            onServiceWorkerRegistrationFailure);
  };

  if (typeof window === 'undefined') {
    serviceWorkerMain();
  } else {
    maybeRegisterServiceWorker();
  }
})();
