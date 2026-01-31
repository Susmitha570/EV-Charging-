const btn = document.getElementById("btn");
const msg = document.getElementById("msg");

const kpiDistance = document.getElementById("kpiDistance");
const kpiDuration = document.getElementById("kpiDuration");
const kpiAvail = document.getElementById("kpiAvail");
const kpiReq = document.getElementById("kpiReq");
const kpiExtra = document.getElementById("kpiExtra");
const kpiInterval = document.getElementById("kpiInterval");

const otherRoutes = document.getElementById("otherRoutes");
const mapFrame = document.getElementById("mapFrame");
const openMap = document.getElementById("openMap");

const predCard = document.getElementById("predCard");
const predBox = document.getElementById("predictions");

function setMsg(text, type="") {
  msg.className = "msg" + (type ? ` ${type}` : "");
  msg.textContent = text;
}

function setLoading(isLoading){
  btn.disabled = isLoading;
  btn.textContent = isLoading ? "Loading..." : "Recommend";
}

function setKPIs(data){
  kpiDistance.textContent = data?.best_route?.distance_km ? `${data.best_route.distance_km.toFixed(2)} km` : "--";
  kpiDuration.textContent = data?.best_route?.duration_min ? `${data.best_route.duration_min.toFixed(1)} min` : "--";
  kpiInterval.textContent = data?.kpis?.interval_km ? `${data.kpis.interval_km} km` : "--";
  kpiReq.textContent = (data?.kpis?.stations_required ?? "--");
  kpiAvail.textContent = (data?.kpis?.stations_available_dataset ?? "--");
  kpiExtra.textContent = (data?.kpis?.extra_needed ?? "--");
}

function renderOtherRoutes(routes){
  otherRoutes.innerHTML = "";
  if(!routes || routes.length === 0){
    otherRoutes.innerHTML = `<div class="routeItem">No routes found</div>`;
    return;
  }
  routes.forEach(r => {
    const div = document.createElement("div");
    div.className = "routeItem";
    div.textContent = `Route ${r.route_no}  •  ${r.distance_km.toFixed(2)} km  •  ${r.duration_min.toFixed(1)} min`;
    otherRoutes.appendChild(div);
  });
}

function renderPredictions(pred){
  if(!pred){
    predCard.style.display = "none";
    predBox.innerHTML = "";
    return;
  }
  predCard.style.display = "block";
  predBox.innerHTML = `
    <div><b>Predicted (Linear Regression):</b> ${pred.predicted_lr_hours.toFixed(2)} hours</div>
    <div><b>Predicted (Random Forest):</b> ${pred.predicted_rf_hours.toFixed(2)} hours</div>
  `;
}

async function recommend(){
  const source_city = document.getElementById("source").value.trim();
  const destination_city = document.getElementById("destination").value.trim();
  const vehicle_type = document.getElementById("vehicle").value.trim();

  if(!source_city || !destination_city || !vehicle_type){
    setMsg("Please enter Source, Destination and Vehicle.", "error");
    return;
  }

  setLoading(true);
  setMsg("Fetching best route from OSRM...", "");

  try {
    const res = await fetch("/api/recommend", {
      method: "POST",
      headers: {"Content-Type":"application/json"},
      body: JSON.stringify({ source_city, destination_city, vehicle_type })
    });

    const data = await res.json();
    if(!data.ok){
      setMsg(data.error || "Error occurred", "error");
      setLoading(false);
      return;
    }

    setMsg("✅ Route generated successfully", "ok");
    setKPIs(data);
    renderOtherRoutes(data.other_routes);

    // map
    if(data.map_url){
      mapFrame.src = data.map_url;
      openMap.href = data.map_url;
    }

    // predictions (optional)
    renderPredictions(data.predictions);

  } catch (e){
    setMsg("Server not reachable. Backend run avuthundaa check cheyyi.", "error");
  } finally {
    setLoading(false);
  }
}

btn.addEventListener("click", recommend);
