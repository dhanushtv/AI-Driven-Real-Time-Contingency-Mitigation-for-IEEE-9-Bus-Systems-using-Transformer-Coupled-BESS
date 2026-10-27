import pandapower as pp
import numpy as np
import pandas as pd

def create_ieee9_network():
    net = pp.create_empty_network()
    
    # --- 1. CREATE BUSES ---
    b1 = pp.create_bus(net, vn_kv=138.0, name="Bus 1")
    b2 = pp.create_bus(net, vn_kv=138.0, name="Bus 2")
    b3 = pp.create_bus(net, vn_kv=138.0, name="Bus 3")
    b4 = pp.create_bus(net, vn_kv=138.0, name="Bus 4")
    b5 = pp.create_bus(net, vn_kv=138.0, name="Bus 5")
    b6 = pp.create_bus(net, vn_kv=138.0, name="Bus 6")
    b7 = pp.create_bus(net, vn_kv=138.0, name="Bus 7")
    b8 = pp.create_bus(net, vn_kv=138.0, name="Bus 8")
    b9 = pp.create_bus(net, vn_kv=138.0, name="Bus 9")
    
    # --- 2. GENERATORS & LOADS ---
    pp.create_ext_grid(net, bus=b1, vm_pu=1.0, name="Slack Grid")
    pp.create_gen(net, bus=b2, p_mw=163.0, vm_pu=1.02, name="Gen 2")
    pp.create_gen(net, bus=b3, p_mw=85.0, vm_pu=1.02, name="Gen 3")
    
    # IEEE 9-Bus Standard Loads
    pp.create_load(net, bus=b5, p_mw=125.0, q_mvar=50.0, name="Load 5")
    pp.create_load(net, bus=b8, p_mw=100.0, q_mvar=35.0, name="Load 8")
    pp.create_load(net, bus=b6, p_mw=90.0, q_mvar=30.0, name="Load 6")
    
    # --- 3. TRANSMISSION LINES ---
    pp.create_line_from_parameters(net, from_bus=b4, to_bus=b5, length_km=10, r_ohm_per_km=0.01, x_ohm_per_km=0.085, c_nf_per_km=0, max_i_ka=1.2, name="Line 4-5")
    pp.create_line_from_parameters(net, from_bus=b5, to_bus=b7, length_km=10, r_ohm_per_km=0.032, x_ohm_per_km=0.161, c_nf_per_km=0, max_i_ka=1.2, name="Line 5-7")
    pp.create_line_from_parameters(net, from_bus=b7, to_bus=b8, length_km=10, r_ohm_per_km=0.0085, x_ohm_per_km=0.072, c_nf_per_km=0, max_i_ka=1.2, name="Line 7-8")
    
    # FIXED: Changed from_bus=b8, to_bus=9 to from_bus=b8, to_bus=b9
    pp.create_line_from_parameters(net, from_bus=b8, to_bus=b9, length_km=10, r_ohm_per_km=0.0119, x_ohm_per_km=0.1008, c_nf_per_km=0, max_i_ka=1.2, name="Line 8-9")
    
    pp.create_line_from_parameters(net, from_bus=b9, to_bus=b6, length_km=10, r_ohm_per_km=0.039, x_ohm_per_km=0.17, c_nf_per_km=0, max_i_ka=1.2, name="Line 9-6")
    pp.create_line_from_parameters(net, from_bus=b6, to_bus=b4, length_km=10, r_ohm_per_km=0.0119, x_ohm_per_km=0.1008, c_nf_per_km=0, max_i_ka=1.2, name="Line 6-4")
    
    # --- 4. TRANSFORMER-COUPLED BESS AT BUS 7 ---
    bess_mv_bus = pp.create_bus(net, vn_kv=6.6, name="BESS_6.6kV_Bus")
    
    pp.create_transformer_from_parameters(
        net, hv_bus=b7, lv_bus=bess_mv_bus, sn_mva=100.0,
        vn_hv_kv=138.0, vn_lv_kv=6.6, vk_percent=10.0, vkr_percent=0.5,
        pfe_kw=0.0, i0_percent=0.0, name="BESS_Transformer"
    )
    
    pp.create_sgen(net, bus=bess_mv_bus, p_mw=0.0, q_mvar=0.0, name="BESS_Unit")
    return net

def calculate_bess_required(net, tripped_line_idx, target_loading_pct=95.0):
    net.line.at[tripped_line_idx, 'in_service'] = False
    pp.runpp(net)
    
    overloaded = net.res_line[net.res_line.loading_percent > 100.0]
    if overloaded.empty:
        return 0.0  
        
    critical_line = overloaded.loading_percent.idxmax()
    p_flow_mw = abs(net.res_line.at[critical_line, 'p_from_mw'])
    i_ka_limit = net.line.at[critical_line, 'max_i_ka']
    vn_kv = net.bus.at[net.line.at[critical_line, 'from_bus'], 'vn_kv']
    mva_limit = i_ka_limit * vn_kv * np.sqrt(3)
    
    # PTDF Calculation via 1 MW perturbation on BESS
    bess_sgen_idx = net.sgen[net.sgen.name == "BESS_Unit"].index[0]
    delta_p = 1.0
    net.sgen.at[bess_sgen_idx, 'p_mw'] += delta_p
    pp.runpp(net)
    
    p_flow_perturbed = abs(net.res_line.at[critical_line, 'p_from_mw'])
    ptdf = (p_flow_perturbed - p_flow_mw) / delta_p
    net.sgen.at[bess_sgen_idx, 'p_mw'] -= delta_p
    
    target_flow_mw = (target_loading_pct / 100.0) * mva_limit
    mw_reduction_needed = p_flow_mw - target_flow_mw
    
    if abs(ptdf) < 1e-5:
        return 0.0
        
    return max(0.0, mw_reduction_needed / abs(ptdf))

# --- GENERATE DATASET ACROSS MULTIPLE LOAD SCENARIOS ---
data_samples = []
load_scalers = np.linspace(0.7, 1.4, 40) 

print("Generating 9-bus dataset samples...")
for scaler in load_scalers:
    for line_to_trip in range(len(create_ieee9_network().line)):
        net = create_ieee9_network()
        net.load['p_mw'] *= scaler
        net.load['q_mvar'] *= scaler
        
        pp.runpp(net)
        load_b5 = net.load.at[0, 'p_mw']
        load_b8 = net.load.at[1, 'p_mw']
        load_b6 = net.load.at[2, 'p_mw']
        
        required_mw = calculate_bess_required(net, tripped_line_idx=line_to_trip)
        
        data_samples.append({
            'load_scale': scaler,
            'bus5_load_mw': load_b5,
            'bus8_load_mw': load_b8,
            'bus6_load_mw': load_b6,
            'tripped_line': line_to_trip,
            'required_bess_mw': required_mw
        })

df = pd.DataFrame(data_samples)
df.to_csv('bess_training_data_9bus.csv', index=False)
print(f"DONE! Created 'bess_training_data_9bus.csv' with {len(df)} samples.")