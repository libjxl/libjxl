if (typeof window === 'undefined') {
  const EMBEDDED = {
    'jxl_decoder.js': '$jxl_decoder.js$',
    'jxl_decoder.worker.js': '$jxl_decoder.worker.js$',
  };
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

  self.addEventListener('fetch', function(event) {
    if (event.request.cache === 'only-if-cached' &&
        event.request.mode !== 'same-origin') {
      return;
    }
    const url = event.request.url;
    // Shortcut for baked-in scripts.
    for (const [key, value] of Object.entries(EMBEDDED)) {
      if (url.endsWith(key)) {
        const headers = new Headers();
        headers.set('Content-Type', 'application/javascript');
        // TODO: refactor
        headers.set('Cross-Origin-Embedder-Policy', 'require-corp');
        headers.set('Cross-Origin-Opener-Policy', 'same-origin');

        event.respondWith(new Response(value, {
          status: 200,
          statusText: 'OK',
          headers: headers,
        }));
        return;
      }
    }
    // Bypass HTML page fetching.
    if (event.request.destination !== 'document') {
      return;
    }

    // Overcome COOP/COEP policies for all the other requests.
    event.respondWith(
        fetch(event.request)
            .then((response) => {
              if (response.status === 0) {
                return response;
              }

              const newHeaders = new Headers(response.headers);
              newHeaders.set('Cross-Origin-Embedder-Policy', 'require-corp');
              newHeaders.set('Cross-Origin-Opener-Policy', 'same-origin');

              return new Response(response.body, {
                status: response.status,
                statusText: response.statusText,
                headers: newHeaders,
              });
            })
            .catch((e) => console.error(e)));
  });

} else {  // typeof window !== 'undefined'
  (() => {
    // You can customize the behavior of this script through a global `coi`
    // variable.
    const coi = {
      shouldRegister: () => true,
      shouldDeregister: () => false,
      doReload: () => window.location.reload(),
      quiet: false,
      ...window.coi
    }

    const n = navigator;
    if (coi.shouldDeregister() && n.serviceWorker &&
        n.serviceWorker.controller) {
      n.serviceWorker.controller.postMessage({type: 'deregister'});
    }

    // If we're already coi: do nothing. Perhaps it's due to this script doing
    // its job, or COOP/COEP are already set from the origin server. Also if the
    // browser has no notion of crossOriginIsolated, just give up here.
    if (window.crossOriginIsolated !== false || !coi.shouldRegister()) return;

    if (!window.isSecureContext) {
      !coi.quiet &&
          console.log(
              'COOP/COEP Service Worker not registered, a secure context is required.');
      return;
    }

    // In some environments (e.g. Chrome incognito mode) this won't be available
    if (n.serviceWorker) {
      n.serviceWorker.register(window.document.currentScript.src)
          .then(
              (registration) => {
                !coi.quiet &&
                    console.log(
                        'COOP/COEP Service Worker registered',
                        registration.scope);

                registration.addEventListener('updatefound', () => {
                  !coi.quiet &&
                      console.log(
                          'Reloading page to make use of updated COOP/COEP Service Worker.');
                  coi.doReload()
                });

                // If the registration is active, but it's not controlling the
                // page
                if (registration.active && !n.serviceWorker.controller) {
                  !coi.quiet &&
                      console.log(
                          'Reloading page to make use of COOP/COEP Service Worker.');
                  coi.doReload()
                }
              },
              (err) => {
                !coi.quiet &&
                    console.error(
                        'COOP/COEP Service Worker failed to register:', err);
              });
    }
  })();
}
