// NutriVision — Cloudflare Worker proxy.
//
// Acts as a permanent public URL in front of the (ephemeral) Cloudflare
// quick-tunnel that exposes the Flask backend running on the institute
// server. The Worker URL never changes; if the upstream tunnel URL rotates,
// just edit the UPSTREAM line below and click "Deploy".
//
// Endpoints served (forwarded as-is):
//   GET  /             -> upstream / (Flask index)
//   GET  /healthz      -> upstream /healthz (JSON liveness)
//   POST /api/predict  -> upstream /api/predict
//   POST /api/suggest  -> upstream /api/suggest
//
// Free tier: 100k requests/day, plenty for a thesis demo.

const UPSTREAM = "https://purchasing-budapest-arrange-leadership.trycloudflare.com";

export default {
  async fetch(request) {
    // CORS preflight — let any origin call us from JS.
    if (request.method === "OPTIONS") {
      return new Response(null, {
        headers: {
          "Access-Control-Allow-Origin":  "*",
          "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
          "Access-Control-Allow-Headers": "*",
          "Access-Control-Max-Age":       "86400",
        },
      });
    }

    const url = new URL(request.url);
    const upstreamUrl = UPSTREAM + url.pathname + url.search;

    // Forward method, headers, body unchanged.
    const upstreamReq = new Request(upstreamUrl, {
      method:   request.method,
      headers:  request.headers,
      body:     request.body,
      redirect: "follow",
    });

    let upstreamResp;
    try {
      upstreamResp = await fetch(upstreamReq);
    } catch (err) {
      return new Response(
        JSON.stringify({
          ok: false,
          error:    "Backend unreachable — institute server or tunnel is down.",
          upstream: UPSTREAM,
          detail:   String(err),
        }),
        {
          status:  502,
          headers: {
            "Content-Type":               "application/json",
            "Access-Control-Allow-Origin": "*",
          },
        },
      );
    }

    // Echo the upstream response, layering CORS headers on top.
    const resp = new Response(upstreamResp.body, upstreamResp);
    resp.headers.set("Access-Control-Allow-Origin",  "*");
    resp.headers.set("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
    resp.headers.set("Access-Control-Allow-Headers", "*");
    return resp;
  },
};
