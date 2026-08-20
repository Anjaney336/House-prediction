(function () {
  "use strict";
  const script = document.currentScript;
  const api = (script.dataset.api || "http://127.0.0.1:8765").replace(/\/$/, "");
  let model = script.dataset.model || null;
  const routing = script.dataset.routing === "true" || !model;
  const tenant = script.dataset.tenant || "public";
  const mount = document.getElementById(script.dataset.mount || "pricepredict-widget") || (() => {
    const node = document.createElement("div"); script.parentNode.insertBefore(node, script.nextSibling); return node;
  })();
  const root = mount.attachShadow ? mount.attachShadow({mode: "open"}) : mount;
  root.innerHTML = `<style>
    :host{--pp-accent:#14b8a6;--pp-bg:#081522;--pp-card:#102536;--pp-text:#eef6ff;--pp-muted:#a8bbca;font:15px Inter,system-ui,sans-serif}
    *{box-sizing:border-box}.card{max-width:680px;background:var(--pp-bg);color:var(--pp-text);border:1px solid #294153;border-radius:20px;padding:24px;box-shadow:0 18px 50px #0017}
    h2{margin:0 0 6px;font-size:25px}.sub,.legal{color:var(--pp-muted);line-height:1.45}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:14px;margin:20px 0}
    label{display:block;font-weight:650;font-size:13px}input,select{width:100%;margin-top:6px;padding:11px;border-radius:9px;border:1px solid #385266;background:#0d1d2b;color:var(--pp-text)}
    button{width:100%;padding:13px;border:0;border-radius:10px;background:var(--pp-accent);color:#05201d;font-weight:800;cursor:pointer}button:disabled{opacity:.55}.result{margin-top:18px;padding:18px;border-radius:12px;background:var(--pp-card)}
    .estimate{font-size:30px;font-weight:800}.error{color:#ff9eaa}.legal{font-size:12px;margin-top:14px}.hidden{display:none}.route{border-bottom:1px solid #294153;padding-bottom:20px;margin-bottom:20px}
  </style><section class="card"><h2>Instant property estimate</h2><div class="sub">Choose a supported market and property type. The published model determines the fields you need.</div>
  <section class="route hidden"><div class="grid route-grid"></div><button type="button" class="load-model">Continue</button></section>
  <form class="hidden"><div class="grid feature-grid"></div><button>Estimate value</button></form><div class="result hidden"></div>
  <div class="legal">Not a legal valuation or guaranteed appraisal. Model-based estimate only.</div></section>`;
  const form=root.querySelector("form"), grid=root.querySelector(".feature-grid"), result=root.querySelector(".result"), button=form.querySelector("button"), routeBox=root.querySelector(".route"), routeGrid=root.querySelector(".route-grid"), loadButton=root.querySelector(".load-model");
  const money=value=>new Intl.NumberFormat(undefined,{maximumFractionDigits:0}).format(value);
  let selection=null;
  function show(html,error=false){result.classList.remove("hidden");result.classList.toggle("error",error);result.innerHTML=html;}
  async function jsonFetch(url,options={}){const response=await fetch(url,options);const body=await response.json();if(!response.ok)throw new Error(body.detail||"Request failed.");return body;}
  function renderFeatures(schema){grid.innerHTML="";schema.features.forEach(feature=>{const label=document.createElement("label");label.textContent=feature.label||feature.name;let input;
    if(feature.vocabulary&&feature.vocabulary.length&&feature.vocabulary.length<=100){input=document.createElement("select");const blank=document.createElement("option");blank.value="";blank.textContent="Select…";input.appendChild(blank);feature.vocabulary.forEach(value=>{const option=document.createElement("option");option.value=value;option.textContent=value;input.appendChild(option);});}
    else{input=document.createElement("input");input.type=feature.dtype==="numeric"?"number":"text";if(feature.dtype==="numeric")input.step="any";}
    input.name=feature.name;input.required=!!feature.required;label.appendChild(input);grid.appendChild(label);});form.classList.remove("hidden");}
  function selector(labelText,name,values){const label=document.createElement("label");label.textContent=labelText;const select=document.createElement("select");select.name=name;[...new Set(values)].forEach(value=>{const option=document.createElement("option");option.value=value;option.textContent=value;select.appendChild(option);});label.appendChild(select);routeGrid.appendChild(label);return select;}
  async function initialize(){try{
    if(!routing){const schema=await jsonFetch(`${api}/schema?model_id=${encodeURIComponent(model)}`,{headers:{"X-Tenant-ID":tenant}});renderFeatures(schema);return;}
    const catalog=await jsonFetch(`${api}/markets`,{headers:{"X-Tenant-ID":tenant}});if(!catalog.length){show("No approved production valuation model is currently available. Please contact the operator.",true);return;}
    routeBox.classList.remove("hidden");const market=selector("Market","market",catalog.map(row=>row.market));const asset=selector("Asset","asset_type",catalog.map(row=>row.asset_type));const property=selector("Property type","property_type",catalog.map(row=>row.property_type));const transaction=selector("Transaction","transaction_type",catalog.map(row=>row.transaction_type));
    loadButton.addEventListener("click",async()=>{try{selection={market:market.value,asset_type:asset.value,property_type:property.value,transaction_type:transaction.value};const query=new URLSearchParams(selection);const routed=await jsonFetch(`${api}/route?${query}`,{headers:{"X-Tenant-ID":tenant}});model=routed.id;renderFeatures(routed.model_card.prediction_contract);result.classList.add("hidden");}catch(error){show(error.message,true);}});
  }catch(error){show(error.message,true);}}
  form.addEventListener("submit",async event=>{event.preventDefault();button.disabled=true;button.textContent="Estimating…";result.classList.add("hidden");const values={};new FormData(form).forEach((value,key)=>{if(value!=="")values[key]=value;});try{
    const endpoint=routing?"/valuation":"/predict";const payload=routing?{...selection,values}:{model_id:model,values};const body=await jsonFetch(`${api}${endpoint}`,{method:"POST",headers:{"Content-Type":"application/json","X-Tenant-ID":tenant},body:JSON.stringify(payload)});
    const caveat=body.aggregate_data_caveat?"<p><strong>Coverage caveat:</strong> this model was trained on aggregate-level observations, not exact properties.</p>":"";const warnings=body.ood.warnings.length?`<p class="legal">Coverage notes: ${body.ood.warnings.join(" ")}</p>`:"";
    show(`<div class="estimate">${body.currency||""} ${money(body.estimate)}</div><div>Calibrated range: ${money(body.range.lower)} – ${money(body.range.upper)}</div><p>Model confidence: <strong>${body.model_confidence.score}/100 · ${body.model_confidence.label}</strong> <span class="legal">(heuristic, not a probability)</span></p><p>${body.market||""} · ${body.property_type||""} · ${body.transaction_type||""}</p>${caveat}${warnings}<p class="legal">Model ${body.model_id}, updated ${body.last_updated||"unknown"}. ${body.disclaimer}</p>`);
  }catch(error){show(error.message,true);}finally{button.disabled=false;button.textContent="Estimate value";}});
  initialize();
})();
