<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Universal RFM Customer Segmentation Engine</title>
    <!-- Tailwind CSS CDN -->
    <script src="https://cdn.tailwindcss.com"></script>
    <!-- Plotly.js CDN -->
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <!-- PapaParse CDN for CSV Handling -->
    <script src="https://cdnjs.cloudflare.com/ajax/libs/PapaParse/5.4.1/papaparse.min.js"></script>
</head>
<body class="bg-gray-50 text-gray-800 font-sans min-h-screen flex flex-col">

    <!-- Header / Navbar -->
    <header class="bg-slate-900 text-white p-6 shadow-md">
        <div class="max-w-7xl mx-auto flex justify-between items-center">
            <div>
                <h1 class="text-2xl font-bold tracking-tight">Universal RFM Customer Segmentation Engine</h1>
                <p class="text-slate-400 text-sm mt-1">A dynamic client-side machine learning engine executing K-Means on the fly in pure JavaScript.</p>
            </div>
        </div>
    </header>

    <div class="max-w-7xl mx-auto w-full p-6 flex-grow grid grid-cols-1 lg:grid-cols-4 gap-6">

        <!-- Sidebar / Control Panel -->
        <aside class="lg:col-span-1 bg-white p-5 rounded-xl shadow-sm border border-gray-200 flex flex-col gap-6">
            <div>
                <h2 class="text-lg font-semibold border-b pb-2 mb-4 text-slate-800">📁 Data Input</h2>
                
                <label class="block text-sm font-medium text-gray-700 mb-2">Upload Clean Transaction CSV</label>
                <input type="file" id="csvFileInput" accept=".csv" class="block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-lg file:border-0 file:text-sm file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100 cursor-pointer border rounded-lg p-1"/>
                
                <div class="relative flex py-3 items-center">
                    <div class="flex-grow border-t border-gray-200"></div>
                    <span class="flex-shrink mx-2 text-gray-400 text-xs uppercase font-bold">OR</span>
                    <div class="flex-grow border-t border-gray-200"></div>
                </div>

                <button id="btnDemoData" class="w-full bg-slate-100 hover:bg-slate-200 text-slate-700 font-medium py-2 px-4 rounded-lg transition duration-150 border border-slate-300 text-sm">
                    🎲 Load Sample Demo Data
                </button>
            </div>

            <div class="border-t pt-4">
                <h2 class="text-lg font-semibold border-b pb-2 mb-4 text-slate-800">⚙️ Model Settings</h2>
                <label class="block text-sm font-medium text-gray-700 mb-1">
                    Select Clusters (K): <span id="kValDisplay" class="font-bold text-blue-600">4</span>
                </label>
                <input type="range" id="sliderK" min="2" max="6" value="4" class="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer">
                <p class="text-xs text-gray-500 mt-2">Adjust K to segment customers into more or fewer behavioral tiers.</p>
            </div>

            <button id="btnProcess" disabled class="w-full bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2.5 px-4 rounded-lg transition duration-150 disabled:opacity-50 disabled:cursor-not-allowed text-sm shadow">
                🚀 Process & Segment Data
            </button>
        </aside>

        <!-- Main Content Area -->
        <main class="lg:col-span-3 flex flex-col gap-6">

            <!-- Dataset Guidelines Accordion -->
            <details class="bg-white border border-blue-100 rounded-xl p-4 shadow-sm group">
                <summary class="font-semibold text-blue-900 cursor-pointer flex justify-between items-center text-sm">
                    <span>📋 Dataset Requirements & Preparation Instructions</span>
                    <span class="group-open:rotate-180 transition-transform">▼</span>
                </summary>
                <ul class="text-xs text-gray-600 mt-3 space-y-1.5 list-disc list-inside border-t pt-3 border-blue-50">
                    <li><strong>No Missing Values:</strong> Ensure there are no empty (<code class="bg-gray-100 px-1 py-0.5 rounded">NaN</code> / <code class="bg-gray-100 px-1 py-0.5 rounded">null</code>) cells in key mapped fields.</li>
                    <li><strong>Positive Numeric Values:</strong> <code class="bg-gray-100 px-1 py-0.5 rounded">UnitPrice</code> and <code class="bg-gray-100 px-1 py-0.5 rounded">Quantity</code> must be positive numbers (> 0) without currency symbols ($).</li>
                    <li><strong>Valid Dates:</strong> Dates should be formatted consistently (e.g., <code class="bg-gray-100 px-1 py-0.5 rounded">YYYY-MM-DD</code>).</li>
                    <li><strong>No Cancellations:</strong> Filter out returns, refunds, or negative transaction quantities beforehand.</li>
                </ul>
            </details>

            <!-- Column Mapping Section -->
            <section id="secMapping" class="hidden bg-white p-5 rounded-xl shadow-sm border border-gray-200">
                <h3 class="text-md font-semibold text-gray-800 mb-3">Map Your Data Columns</h3>
                <div class="grid grid-cols-1 sm:grid-cols-3 lg:grid-cols-5 gap-3">
                    <div>
                        <label class="block text-xs font-medium text-gray-600 mb-1">Customer ID</label>
                        <select id="mapId" class="w-full text-xs border border-gray-300 rounded-lg p-2 bg-white focus:ring-blue-500 focus:border-blue-500"></select>
                    </div>
                    <div>
                        <label class="block text-xs font-medium text-gray-600 mb-1">Order Date</label>
                        <select id="mapDate" class="w-full text-xs border border-gray-300 rounded-lg p-2 bg-white focus:ring-blue-500 focus:border-blue-500"></select>
                    </div>
                    <div>
                        <label class="block text-xs font-medium text-gray-600 mb-1">Order ID</label>
                        <select id="mapOrder" class="w-full text-xs border border-gray-300 rounded-lg p-2 bg-white focus:ring-blue-500 focus:border-blue-500"></select>
                    </div>
                    <div>
                        <label class="block text-xs font-medium text-gray-600 mb-1">Unit Price</label>
                        <select id="mapPrice" class="w-full text-xs border border-gray-300 rounded-lg p-2 bg-white focus:ring-blue-500 focus:border-blue-500"></select>
                    </div>
                    <div>
                        <label class="block text-xs font-medium text-gray-600 mb-1">Quantity</label>
                        <select id="mapQty" class="w-full text-xs border border-gray-300 rounded-lg p-2 bg-white focus:ring-blue-500 focus:border-blue-500"></select>
                    </div>
                </div>
            </section>

            <!-- Error Banner -->
            <div id="alertBox" class="hidden p-4 rounded-xl text-sm font-medium border"></div>

            <!-- Tabbed Interface for Results & Diagnostics -->
            <div id="secResults" class="hidden flex-col gap-6">
                <!-- Navigation Tabs -->
                <div class="border-b border-gray-200 bg-white rounded-t-xl px-4 pt-2">
                    <nav class="flex gap-6">
                        <button id="tabBtnDash" class="pb-3 px-1 text-sm font-semibold border-b-2 border-blue-600 text-blue-600">📊 Segmentation Dashboards</button>
                        <button id="tabBtnOpt" class="pb-3 px-1 text-sm font-semibold text-gray-500 border-b-2 border-transparent hover:text-gray-700">📐 Cluster Optimization (Elbow & Silhouette)</button>
                    </nav>
                </div>

                <!-- Tab 1: Segmentation Results -->
                <div id="tabContentDash" class="flex flex-col gap-6">
                    <!-- Key KPI Metrics -->
                    <div class="grid grid-cols-2 md:grid-cols-4 gap-4">
                        <div class="bg-white p-4 rounded-xl border shadow-sm">
                            <p class="text-xs text-gray-500 font-medium">Total Customers</p>
                            <p id="metricTotalCust" class="text-xl font-bold text-gray-800 mt-1">-</p>
                        </div>
                        <div class="bg-white p-4 rounded-xl border shadow-sm">
                            <p class="text-xs text-gray-500 font-medium">Avg Recency (Days)</p>
                            <p id="metricAvgRecency" class="text-xl font-bold text-gray-800 mt-1">-</p>
                        </div>
                        <div class="bg-white p-4 rounded-xl border shadow-sm">
                            <p class="text-xs text-gray-500 font-medium">Avg Frequency (Orders)</p>
                            <p id="metricAvgFreq" class="text-xl font-bold text-gray-800 mt-1">-</p>
                        </div>
                        <div class="bg-white p-4 rounded-xl border shadow-sm">
                            <p class="text-xs text-gray-500 font-medium">Avg Spend ($)</p>
                            <p id="metricAvgSpend" class="text-xl font-bold text-gray-800 mt-1">-</p>
                        </div>
                    </div>

                    <!-- Plotly Visualization Charts -->
                    <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                        <div class="bg-white p-4 rounded-xl border shadow-sm">
                            <h4 class="text-sm font-semibold text-gray-700 mb-2">Customer Segment Distribution</h4>
                            <div id="plotPie" class="w-full h-80"></div>
                        </div>
                        <div class="bg-white p-4 rounded-xl border shadow-sm">
                            <h4 class="text-sm font-semibold text-gray-700 mb-2">Frequency vs Spend (Log Scale)</h4>
                            <div id="plotScatter" class="w-full h-80"></div>
                        </div>
                    </div>

                    <!-- Output Data Table -->
                    <div class="bg-white p-5 rounded-xl border shadow-sm">
                        <div class="flex justify-between items-center mb-4">
                            <h4 class="text-sm font-semibold text-gray-700">Segmented Customer Table</h4>
                            <button id="btnDownload" class="bg-green-600 hover:bg-green-700 text-white text-xs font-semibold py-1.5 px-3 rounded-lg shadow transition">
                                📥 Download Segmented Results (CSV)
                            </button>
                        </div>
                        <div class="max-h-64 overflow-y-auto border rounded-lg">
                            <table class="w-full text-left text-xs border-collapse">
                                <thead class="bg-gray-100 sticky top-0 border-b">
                                    <tr>
                                        <th class="p-2.5 font-semibold text-gray-700">Customer ID</th>
                                        <th class="p-2.5 font-semibold text-gray-700">Customer Segment</th>
                                        <th class="p-2.5 font-semibold text-gray-700">Recency (Days)</th>
                                        <th class="p-2.5 font-semibold text-gray-700">Frequency (Orders)</th>
                                        <th class="p-2.5 font-semibold text-gray-700">Monetary ($)</th>
                                    </tr>
                                </thead>
                                <tbody id="tableBody" class="divide-y"></tbody>
                            </table>
                        </div>
                    </div>
                </div>

                <!-- Tab 2: Optimization Diagnostics -->
                <div id="tabContentOpt" class="hidden flex-col gap-6">
                    <div class="bg-white p-4 rounded-xl border shadow-sm">
                        <h3 class="text-md font-semibold text-gray-800">Determine Optimal Number of Clusters</h3>
                        <p class="text-xs text-gray-500 mt-1">Use these diagnostic charts to evaluate the mathematical structure of your dataset.</p>
                    </div>

                    <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                        <div class="bg-white p-4 rounded-xl border shadow-sm">
                            <h4 class="text-sm font-semibold text-gray-700 mb-2">1. Elbow Method (WCSS)</h4>
                            <div id="plotElbow" class="w-full h-72"></div>
                            <div class="bg-blue-50 p-3 rounded-lg text-xs text-blue-800 border border-blue-100 mt-2">
                                💡 <strong>How to read:</strong> Look for the "bend" or elbow point where WCSS drops off significantly less.
                            </div>
                        </div>
                        <div class="bg-white p-4 rounded-xl border shadow-sm">
                            <h4 class="text-sm font-semibold text-gray-700 mb-2">2. Silhouette Analysis</h4>
                            <div id="plotSilhouette" class="w-full h-72"></div>
                            <div id="silhouetteWinnerBox" class="bg-green-50 p-3 rounded-lg text-xs text-green-800 border border-green-200 mt-2"></div>
                        </div>
                    </div>
                </div>
            </div>
        </main>
    </div>

    <!-- Footer -->
    <footer class="bg-white border-t p-4 text-center text-xs text-gray-500 mt-auto">
        © 2026 • Built with Pure JS, Plotly & Tailwind • All Rights Reserved
    </footer>

    <!-- CLIENT-SIDE APPLICATION LOGIC -->
    <script>
        // --- GLOBAL STATE ---
        let rawDataset = null;
        let processedRfmResults = null;

        // --- DOM ELEMENTS ---
        const csvFileInput = document.getElementById('csvFileInput');
        const btnDemoData = document.getElementById('btnDemoData');
        const sliderK = document.getElementById('sliderK');
        const kValDisplay = document.getElementById('kValDisplay');
        const btnProcess = document.getElementById('btnProcess');
        const secMapping = document.getElementById('secMapping');
        const secResults = document.getElementById('secResults');
        const alertBox = document.getElementById('alertBox');

        const mapId = document.getElementById('mapId');
        const mapDate = document.getElementById('mapDate');
        const mapOrder = document.getElementById('mapOrder');
        const mapPrice = document.getElementById('mapPrice');
        const mapQty = document.getElementById('mapQty');

        // --- EVENT LISTENERS ---
        sliderK.addEventListener('input', (e) => kValDisplay.textContent = e.target.value);

        btnDemoData.addEventListener('click', () => {
            rawDataset = generateSyntheticData(1000, 150);
            populateMappingDropdowns();
            btnProcess.disabled = false;
            hideAlert();
        });

        csvFileInput.addEventListener('change', (e) => {
            const file = e.target.files[0];
            if (!file) return;

            Papa.parse(file, {
                header: true,
                skipEmptyLines: true,
                complete: (results) => {
                    if (results.data && results.data.length > 0) {
                        rawDataset = results.data;
                        populateMappingDropdowns();
                        btnProcess.disabled = false;
                        hideAlert();
                    } else {
                        showAlert("❌ Could not parse any valid records from the uploaded CSV.", "red");
                    }
                },
                error: (err) => showAlert(`❌ CSV Parse Error: ${err.message}`, "red")
            });
        });

        btnProcess.addEventListener('click', () => runPipeline());

        // Tab Switching Logic
        const tabBtnDash = document.getElementById('tabBtnDash');
        const tabBtnOpt = document.getElementById('tabBtnOpt');
        const tabContentDash = document.getElementById('tabContentDash');
        const tabContentOpt = document.getElementById('tabContentOpt');

        tabBtnDash.addEventListener('click', () => {
            tabBtnDash.className = "pb-3 px-1 text-sm font-semibold border-b-2 border-blue-600 text-blue-600";
            tabBtnOpt.className = "pb-3 px-1 text-sm font-semibold text-gray-500 border-b-2 border-transparent hover:text-gray-700";
            tabContentDash.classList.remove('hidden');
            tabContentDash.classList.add('flex');
            tabContentOpt.classList.add('hidden');
            tabContentOpt.classList.remove('flex');
        });

        tabBtnOpt.addEventListener('click', () => {
            tabBtnOpt.className = "pb-3 px-1 text-sm font-semibold border-b-2 border-blue-600 text-blue-600";
            tabBtnDash.className = "pb-3 px-1 text-sm font-semibold text-gray-500 border-b-2 border-transparent hover:text-gray-700";
            tabContentOpt.classList.remove('hidden');
            tabContentOpt.classList.add('flex');
            tabContentDash.classList.add('hidden');
            tabContentDash.classList.remove('flex');
            Plotly.Plots.resize('plotElbow');
            Plotly.Plots.resize('plotSilhouette');
        });

        document.getElementById('btnDownload').addEventListener('click', () => downloadCSV());

        // --- HELPER FUNCTIONS ---
        function showAlert(msg, color = "red") {
            alertBox.className = `p-4 rounded-xl text-sm font-medium border bg-${color}-50 text-${color}-800 border-${color}-200 block`;
            alertBox.innerHTML = msg;
        }

        function hideAlert() {
            alertBox.className = "hidden";
        }

        function findDefaultCol(candidates, columns) {
            for (let col of columns) {
                if (candidates.some(cand => col.toLowerCase().includes(cand))) {
                    return col;
                }
            }
            return columns[0] || '';
        }

        function populateMappingDropdowns() {
            if (!rawDataset || rawDataset.length === 0) return;
            const cols = Object.keys(rawDataset[0]);

            const dropdowns = [
                { el: mapId, cand: ["cust", "user", "client"] },
                { el: mapDate, cand: ["date", "time", "day"] },
                { el: mapOrder, cand: ["order", "txn", "invoice"] },
                { el: mapPrice, cand: ["price", "rate", "cost"] },
                { el: mapQty, cand: ["qty", "quantity", "count"] }
            ];

            dropdowns.forEach(({ el, cand }) => {
                el.innerHTML = '';
                cols.forEach(col => {
                    const opt = document.createElement('option');
                    opt.value = col;
                    opt.textContent = col;
                    el.appendChild(opt);
                });
                el.value = findDefaultCol(cand, cols);
            });

            secMapping.classList.remove('hidden');
        }

        // --- SYNTHETIC DATA GENERATOR ---
        function generateSyntheticData(numRows = 1000, numCust = 150) {
            const custIds = Array.from({ length: numCust }, (_, i) => `CUST-${1000 + i}`);
            const orderIds = Array.from({ length: 300 }, (_, i) => `INV-${50000 + i}`);
            const rows = [];
            
            const baseTime = new Date('2026-06-30T00:00:00').getTime();

            for (let i = 0; i < numRows; i++) {
                const randomDays = Math.floor(Math.random() * 365);
                const orderDate = new Date(baseTime - randomDays * 86400000).toISOString().split('T')[0];
                const unitPrice = (Math.exp(Math.random() * 3) + 1.5).toFixed(2);
                const qty = Math.floor(Math.random() * 11) + 1;

                rows.push({
                    CustomerID: custIds[Math.floor(Math.random() * custIds.length)],
                    OrderDate: orderDate,
                    OrderID: orderIds[Math.floor(Math.random() * orderIds.length)],
                    UnitPrice: parseFloat(unitPrice),
                    Quantity: qty
                });
            }
            return rows;
        }

        // --- CORE RFM & K-MEANS ALGORITHMS ---
        function runPipeline() {
            hideAlert();

            const idCol = mapId.value;
            const dateCol = mapDate.value;
            const orderCol = mapOrder.value;
            const priceCol = mapPrice.value;
            const qtyCol = mapQty.value;

            const selectedCols = [idCol, dateCol, orderCol, priceCol, qtyCol];
            if (new Set(selectedCols).size < 5) {
                showAlert("⚠️ <strong>Mapping Error:</strong> Please select 5 unique columns. Two or more dropdowns point to the same header.");
                return;
            }

            // 1. Data Ingestion & Cleaning
            const cleanRows = [];
            for (let row of rawDataset) {
                const cId = String(row[idCol] || '').trim();
                const dVal = new Date(row[dateCol]);
                const oId = String(row[orderCol] || '').trim();
                const price = parseFloat(row[priceCol]);
                const qty = parseFloat(row[qtyCol]);

                if (!cId || isNaN(dVal.getTime()) || !oId || isNaN(price) || isNaN(qty)) {
                    showAlert(`❌ <strong>Data Ingestion Error:</strong> Missing or malformed values detected in mapped columns.`, "red");
                    return;
                }

                cleanRows.push({
                    CustomerID: cId,
                    OrderDate: dVal,
                    OrderID: oId,
                    UnitPrice: price,
                    Quantity: qty,
                    SalesAmount: price * qty
                });
            }

            // 2. Aggregate RFM Metrics
            let maxTime = 0;
            cleanRows.forEach(r => { if (r.OrderDate.getTime() > maxTime) maxTime = r.OrderDate.getTime(); });
            const refDate = maxTime + 86400000;

            const customerMap = {};
            cleanRows.forEach(r => {
                if (!customerMap[r.CustomerID]) {
                    customerMap[r.CustomerID] = { CustomerID: r.CustomerID, maxDate: r.OrderDate.getTime(), orders: new Set(), sales: 0 };
                }
                const c = customerMap[r.CustomerID];
                if (r.OrderDate.getTime() > c.maxDate) c.maxDate = r.OrderDate.getTime();
                c.orders.add(r.OrderID);
                c.sales += r.SalesAmount;
            });

            const rfmList = Object.values(customerMap).map(c => ({
                CustomerID: c.CustomerID,
                Recency: Math.round((refDate - c.maxDate) / 86400000),
                Frequency: c.orders.size,
                Monetary: c.sales
            }));

            // 3. Log1p Transformation & Standardization
            const featureMatrix = rfmList.map(r => [
                Math.log1p(r.Recency),
                Math.log1p(r.Frequency),
                Math.log1p(r.Monetary)
            ]);

            const means = [0, 0, 0];
            const stds = [0, 0, 0];
            const n = featureMatrix.length;

            for (let i = 0; i < 3; i++) {
                means[i] = featureMatrix.reduce((sum, row) => sum + row[i], 0) / n;
                const variance = featureMatrix.reduce((sum, row) => sum + Math.pow(row[i] - means[i], 2), 0) / n;
                stds[i] = Math.sqrt(variance) || 1e-8;
            }

            const scaledMatrix = featureMatrix.map(row => row.map((val, j) => (val - means[j]) / stds[j]));

            // 4. Compute Optimization Metrics (Elbow & Silhouette)
            const kSelected = parseInt(sliderK.value);
            const maxK = 8;
            const elbowData = [];
            const silhouetteData = [];

            for (let k = 1; k <= maxK; k++) {
                const res = runKMeans(scaledMatrix, k);
                elbowData.push({ k, wcss: res.wcss });
                if (k >= 2) {
                    const silScore = calculateSilhouetteScore(scaledMatrix, res.assignments, res.centroids);
                    silhouetteData.push({ k, score: silScore });
                }
            }

            // 5. Fit KMeans for Selected K
            const finalKMeans = runKMeans(scaledMatrix, kSelected);
            rfmList.forEach((r, idx) => r.Cluster = finalKMeans.assignments[idx]);

            // 6. Dynamic Persona Mapping based on Monetary rank
            const clusterMonetarySums = Array(kSelected).fill(0);
            const clusterCounts = Array(kSelected).fill(0);

            rfmList.forEach(r => {
                clusterMonetarySums[r.Cluster] += r.Monetary;
                clusterCounts[r.Cluster]++;
            });

            const clusterMeans = clusterMonetarySums.map((sum, cIdx) => ({
                cluster: cIdx,
                avgMonetary: sum / (clusterCounts[cIdx] || 1)
            })).sort((a, b) => a.avgMonetary - b.avgMonetary);

            let personaLabels = [];
            if (kSelected === 4) {
                personaLabels = [
                    "Lost / Very Lapsed Customers",
                    "Hibernating / Slipping Customers",
                    "Warm Leads / Potential Churners",
                    "Champions / High-Value Loyal Customers"
                ];
            } else if (kSelected === 3) {
                personaLabels = [
                    "Lost / Low-Value Customers",
                    "Mid-Tier / Warm Leads",
                    "Champions / High-Value Loyal Customers"
                ];
            } else if (kSelected === 2) {
                personaLabels = ["Low-Value / Inactive", "High-Value / Active"];
            } else {
                personaLabels = Array.from({ length: kSelected }, (_, i) => `Segment Tier ${i + 1} (Avg Spend: Low-to-High)`);
            }

            const personaMap = {};
            clusterMeans.forEach((item, rank) => {
                personaMap[item.cluster] = personaLabels[rank];
            });

            rfmList.forEach(r => r.Segment = personaMap[r.Cluster]);

            processedRfmResults = rfmList;

            // 7. Render Frontend Dashboards
            renderDashboards(rfmList, elbowData, silhouetteData, kSelected);
            secResults.classList.remove('hidden');
            secResults.classList.add('flex');
        }

        // --- K-MEANS ENGINE IMPLEMENTATION ---
        function runKMeans(X, k, maxIter = 100) {
            const numSamples = X.length;
            const numFeatures = X[0].length;

            // Deterministic centroid initialization
            let centroids = [];
            for (let i = 0; i < k; i++) {
                const idx = Math.floor((i * numSamples) / k);
                centroids.push([...X[idx]]);
            }

            let assignments = new Array(numSamples).fill(0);
            let wcss = 0;

            for (let iter = 0; iter < maxIter; iter++) {
                let changed = false;
                wcss = 0;

                for (let i = 0; i < numSamples; i++) {
                    let minDist = Infinity;
                    let bestCluster = 0;

                    for (let c = 0; c < k; c++) {
                        let dist = 0;
                        for (let j = 0; j < numFeatures; j++) {
                            dist += Math.pow(X[i][j] - centroids[c][j], 2);
                        }
                        if (dist < minDist) {
                            minDist = dist;
                            bestCluster = c;
                        }
                    }

                    wcss += minDist;
                    if (assignments[i] !== bestCluster) {
                        assignments[i] = bestCluster;
                        changed = true;
                    }
                }

                if (!changed && iter > 0) break;

                const newCentroids = Array.from({ length: k }, () => new Array(numFeatures).fill(0));
                const counts = new Array(k).fill(0);

                for (let i = 0; i < numSamples; i++) {
                    const c = assignments[i];
                    counts[c]++;
                    for (let j = 0; j < numFeatures; j++) {
                        newCentroids[c][j] += X[i][j];
                    }
                }

                for (let c = 0; c < k; c++) {
                    if (counts[c] > 0) {
                        for (let j = 0; j < numFeatures; j++) {
                            newCentroids[c][j] /= counts[c];
                        }
                    } else {
                        newCentroids[c] = [...X[Math.floor(Math.random() * numSamples)]];
                    }
                }
                centroids = newCentroids;
            }

            return { centroids, assignments, wcss };
        }

        // --- SILHOUETTE SCORE CALCULATION ---
        function calculateSilhouetteScore(X, assignments, centroids) {
            const n = X.length;
            if (n < 2) return 0;

            let totalSil = 0;

            for (let i = 0; i < n; i++) {
                const ownCluster = assignments[i];
                let a = 0;
                let ownCount = 0;

                const otherClusterDists = {};

                for (let j = 0; j < n; j++) {
                    let d = 0;
                    for (let f = 0; f < 3; f++) d += Math.pow(X[i][f] - X[j][f], 2);
                    d = Math.sqrt(d);

                    if (assignments[j] === ownCluster) {
                        a += d;
                        ownCount++;
                    } else {
                        const otherC = assignments[j];
                        if (!otherClusterDists[otherC]) otherClusterDists[otherC] = { sum: 0, count: 0 };
                        otherClusterDists[otherC].sum += d;
                        otherClusterDists[otherC].count++;
                    }
                }

                a = ownCount > 1 ? a / (ownCount - 1) : 0;

                let b = Infinity;
                Object.keys(otherClusterDists).forEach(cKey => {
                    const avgD = otherClusterDists[cKey].sum / otherClusterDists[cKey].count;
                    if (avgD < b) b = avgD;
                });

                if (b === Infinity) b = 0;

                const maxAB = Math.max(a, b);
                const s = maxAB > 0 ? (b - a) / maxAB : 0;
                totalSil += s;
            }

            return totalSil / n;
        }

        // --- RENDERING CHARTS & TABLES ---
        function renderDashboards(rfmList, elbowData, silhouetteData, kSelected) {
            // Metrics Calculations
            const totalCust = rfmList.length;
            const avgRec = (rfmList.reduce((acc, r) => acc + r.Recency, 0) / totalCust).toFixed(1);
            const avgFreq = (rfmList.reduce((acc, r) => acc + r.Frequency, 0) / totalCust).toFixed(1);
            const avgSpend = (rfmList.reduce((acc, r) => acc + r.Monetary, 0) / totalCust).toFixed(2);

            document.getElementById('metricTotalCust').textContent = totalCust.toLocaleString();
            document.getElementById('metricAvgRecency').textContent = avgRec;
            document.getElementById('metricAvgFreq').textContent = avgFreq;
            document.getElementById('metricAvgSpend').textContent = `$${parseFloat(avgSpend).toLocaleString()}`;

            // Segment Distribution Donut Chart
            const segmentCounts = {};
            rfmList.forEach(r => segmentCounts[r.Segment] = (segmentCounts[r.Segment] || 0) + 1);

            Plotly.newPlot('plotPie', [{
                labels: Object.keys(segmentCounts),
                values: Object.values(segmentCounts),
                type: 'pie',
                hole: 0.4,
                marker: { colors: ['#66c2a5','#fc8d62','#8da0cb','#e78ac3','#a6d854','#ffd92f'] }
            }], { margin: { t: 10, b: 10, l: 10, r: 10 } }, { responsive: true });

            // Scatter Plot: Frequency vs Spend
            const segments = Array.from(new Set(rfmList.map(r => r.Segment)));
            const scatterTraces = segments.map(seg => {
                const sub = rfmList.filter(r => r.Segment === seg);
                return {
                    x: sub.map(r => r.Frequency),
                    y: sub.map(r => r.Monetary),
                    text: sub.map(r => `Customer: ${r.CustomerID}<br>Recency: ${r.Recency} days`),
                    mode: 'markers',
                    name: seg,
                    marker: { size: sub.map(r => Math.max(6, 20 - r.Recency / 20)) }
                };
            });

            Plotly.newPlot('plotScatter', scatterTraces, {
                xaxis: { title: 'Frequency (Orders)', type: 'log' },
                yaxis: { title: 'Monetary ($)', type: 'log' },
                margin: { t: 10, b: 40, l: 50, r: 10 }
            }, { responsive: true });

            // Table Rendering
            const tbody = document.getElementById('tableBody');
            tbody.innerHTML = '';
            rfmList.slice(0, 100).forEach(r => {
                const tr = document.createElement('tr');
                tr.innerHTML = `
                    <td class="p-2.5 font-medium text-gray-800">${r.CustomerID}</td>
                    <td class="p-2.5"><span class="bg-blue-50 text-blue-700 font-semibold px-2 py-0.5 rounded-full border border-blue-200">${r.Segment}</span></td>
                    <td class="p-2.5">${r.Recency}</td>
                    <td class="p-2.5">${r.Frequency}</td>
                    <td class="p-2.5 font-medium">$${r.Monetary.toFixed(2)}</td>
                `;
                tbody.appendChild(tr);
            });

            // Elbow Plot
            Plotly.newPlot('plotElbow', [{
                x: elbowData.map(d => d.k),
                y: elbowData.map(d => d.wcss),
                type: 'scatter',
                mode: 'lines+markers',
                marker: { color: '#2563eb', size: 8 },
                line: { shape: 'spline' }
            }], {
                xaxis: { title: 'Number of Clusters (K)' },
                yaxis: { title: 'WCSS' },
                shapes: [{ type: 'line', x0: kSelected, x1: kSelected, y0: 0, y1: 1, yref: 'paper', line: { color: 'red', dash: 'dash' } }],
                margin: { t: 10, b: 40, l: 50, r: 10 }
            }, { responsive: true });

            // Silhouette Score Chart
            let bestKObj = { k: 2, score: -1 };
            silhouetteData.forEach(d => { if (d.score > bestKObj.score) bestKObj = d; });

            Plotly.newPlot('plotSilhouette', [{
                x: silhouetteData.map(d => d.k),
                y: silhouetteData.map(d => d.score),
                type: 'bar',
                marker: { color: '#60a5fa' }
            }], {
                xaxis: { title: 'Number of Clusters (K)' },
                yaxis: { title: 'Silhouette Score' },
                shapes: [{ type: 'line', x0: kSelected, x1: kSelected, y0: 0, y1: 1, yref: 'paper', line: { color: 'red', dash: 'dash' } }],
                margin: { t: 10, b: 40, l: 50, r: 10 }
            }, { responsive: true });

            document.getElementById('silhouetteWinnerBox').innerHTML = `
                🏆 <strong>Highest Silhouette Score:</strong> K = ${bestKObj.k} (Score: ${bestKObj.score.toFixed(3)}). 
                This indicates the cleanest cluster separation for this dataset.
            `;
        }

        // CSV Export Helper
        function downloadCSV() {
            if (!processedRfmResults || processedRfmResults.length === 0) return;
            
            const headers = ["CustomerID", "Customer Segment", "Recency", "Frequency", "Monetary"];
            const csvRows = [headers.join(",")];

            processedRfmResults.forEach(r => {
                csvRows.push([
                    `"${r.CustomerID}"`,
                    `"${r.Segment}"`,
                    r.Recency,
                    r.Frequency,
                    r.Monetary.toFixed(2)
                ].join(","));
            });

            const blob = new Blob([csvRows.join("\n")], { type: "text/csv;charset=utf-8;" });
            const url = URL.createObjectURL(blob);
            const a = document.createElement("a");
            a.setAttribute("href", url);
            a.setAttribute("download", "rfm_segmented_customers.csv");
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
        }
    </script>
</body>
</html>
