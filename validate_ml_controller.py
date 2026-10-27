import pandapower as pp
import numpy as np
import joblib
import time

def create_ieee9_network_peak():
    net = pp.create_empty_network()
    b1 = pp.create_bus(net, vn_kv=138.0, name="Bus 1")
    b2 = pp.create_bus(net, vn_kv=138.0, name="Bus 2")
    b3 = pp.create_bus(net, vn_kv=138.0, name="Bus 3")
    b4 = pp.create_bus(net, vn_kv=138.0, name="Bus 4")
    b5 = pp.create_bus(net, vn_kv=138.0, name="Bus 5")
    b6 = pp.create_bus(net, vn_kv=138.0, name="Bus 6")
    b7 = pp.create_bus(net, vn_kv=138.0, name="Bus 7")
    b8 = pp.create_bus(net, vn_kv=138.0, name="Bus 8")
    b9 = pp.create_bus(net, vn_kv=138.0, name="Bus 9")
    
    pp.create_ext_grid(net, bus=b1, vm_pu=1.0)
    pp.create_gen(net, bus=b2, p_mw=163.0, vm_pu=1.02)
    pp.create_gen(net, bus=b3, p_mw=85.0, vm_pu=1.02)
    
    # Increased scaler to 1.35 to reliably induce thermal overloads
    scaler = 1.35
    pp.create_load(net, bus=b5, p_mw=125.0 * scaler, q_mvar=50.0 * scaler)
    pp.create_load(net, bus=b8, p_mw=100.0 * scaler, q_mvar=35.0 * scaler)
    pp.create_load(net, bus=b6, p_mw=90.0 * scaler, q_mvar=30.0 * scaler)
    
    # Tightened current limits (max_i_ka) slightly to demonstrate realistic congestion
    pp.create_line_from_parameters(net, from_bus=b4, to_bus=b5, length_km=10, r_ohm_per_km=0.01, x_ohm_per_km=0.085, c_nf_per_km=0, max_i_ka=1.0, name="Line 4-5")
    pp.create_line_from_parameters(net, from_bus=b5, to_bus=b7, length_km=10, r_ohm_per_km=0.032, x_ohm_per_km=0.161, c_nf_per_km=0, max_i_ka=1.0, name="Line 5-7")
    pp.create_line_from_parameters(net, from_bus=b7, to_bus=b8, length_km=10, r_ohm_per_km=0.0085, x_ohm_per_km=0.072, c_nf_per_km=0, max_i_ka=1.0, name="Line 7-8")
    pp.create_line_from_parameters(net, from_bus=b8, to_bus=b9, length_km=10, r_ohm_per_km=0.0119, x_ohm_per_km=0.1008, c_nf_per_km=0, max_i_ka=1.0, name="Line 8-9")
    pp.create_line_from_parameters(net, from_bus=b9, to_bus=b6, length_km=10, r_ohm_per_km=0.039, x_ohm_per_km=0.17, c_nf_per_km=0, max_i_ka=1.0, name="Line 9-6")
    pp.create_line_from_parameters(net, from_bus=b6, to_bus=b4, length_km=10, r_ohm_per_km=0.0119, x_ohm_per_km=0.1008, c_nf_per_km=0, max_i_ka=1.0, name="Line 6-4")
    
    bess_mv_bus = pp.create_bus(net, vn_kv=6.6, name="BESS_6.6kV_Bus")
    pp.create_transformer_from_parameters(
        net, hv_bus=b7, lv_bus=bess_mv_bus, sn_mva=100.0,
        vn_hv_kv=138.0, vn_lv_kv=6.6, vk_percent=10.0, vkr_percent=0.5,
        pfe_kw=0.0, i0_percent=0.0, name="BESS_Transformer"
    )
    pp.create_sgen(net, bus=bess_mv_bus, p_mw=0.0, q_mvar=0.0, name="BESS_Unit")
    return net, scaler

# 1. Initialize Network and Load Trained ML Model
net, load_scale = create_ieee9_network_peak()
ml_model = joblib.load('bess_ml_controller_9bus.pkl')

# 2. Automatically find a line trip that causes an active overload (> 100%)
tripped_line = None
critical_line = None
initial_loading = 0

for line_idx in range(len(net.line)):
    net_test = create_ieee9_network_peak()[0]
    net_test.line.at[line_idx, 'in_service'] = False
    pp.runpp(net_test)
    overloaded = net_test.res_line[net_test.res_line.loading_percent > 100.0]
    if not overloaded.empty:
        tripped_line = line_idx
        critical_line = overloaded.loading_percent.idxmax()
        initial_loading = overloaded.at[critical_line, 'loading_percent']
        break

if tripped_line is None:
    print("\n[ERROR] No line trips caused an overload. Try increasing the load scaler further.")
else:
    print("\n====================================================")
    print("     9-BUS REAL-TIME ML CONTROL VALIDATION          ")
    print("====================================================")
    print(f"TRIPPED LINE     : Line {tripped_line}")
    print(f"CRITICAL OVERLOAD: Line {critical_line} at {initial_loading:.2f}%")

    # 3. Prepare Feature Vector [load_scale, bus5_load_mw, bus8_load_mw, bus6_load_mw, tripped_line]
    b5_mw = net.load.at[0, 'p_mw']
    b8_mw = net.load.at[1, 'p_mw']
    b6_mw = net.load.at[2, 'p_mw']
    features = [[load_scale, b5_mw, b8_mw, b6_mw, tripped_line]]

    # 4. Predict BESS Injection using ML Model & Measure Response Time
    start_time = time.time()
    predicted_bess_mw = ml_model.predict(features)[0]
    inference_time_ms = (time.time() - start_time) * 1000

    print(f"\nML PREDICTED INJECTION: {predicted_bess_mw:.2f} MW")
    print(f"ML INFERENCE TIME     : {inference_time_ms:.4f} ms")

    # 5. Apply ML Prediction to Grid & Verify AC Power Flow
    net.line.at[tripped_line, 'in_service'] = False # Re-apply trip on main working net
    bess_sgen_idx = net.sgen[net.sgen.name == "BESS_Unit"].index[0]
    net.sgen.at[bess_sgen_idx, 'p_mw'] = predicted_bess_mw
    pp.runpp(net)

    final_loading = net.res_line.at[critical_line, 'loading_percent']
    print(f"\nPOST-CONTROL LOADING  : Line {critical_line} dropped to {final_loading:.2f}%")
    print("====================================================")