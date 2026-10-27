import pandapower as pp
import pandapower.networks as nw
import numpy as np

def run_ieee9_bess_simulation():
    # 1. Load standard IEEE 9-bus system
    net = nw.case9()
    print("--- Initial IEEE 9-Bus System Loaded ---")
    
    # Target high-voltage transmission node for BESS integration
    hv_bus = 7 
    vn_hv = net.bus.loc[hv_bus, 'vn_kv']  # Extracts the high-voltage rating of Bus 7
    print(f"Target Integration Node: Bus {hv_bus} (Voltage Level: {vn_hv} kV)")

    # 2. Create a local 6.6 kV Medium-Voltage Collector Bus for the BESS
    mv_bus = pp.create_bus(net, vn_kv=6.6, name="BESS_6.6kV_Collector_Bus")
    print(f"Created local MV collector bus at ID: {mv_bus} (6.6 kV)")

    # 3. Add the Step-Up Transformer (Coupling 6.6 kV MV bus to HV Transmission Bus 7)
    pp.create_transformer_from_parameters(
        net, 
        hv_bus=hv_bus, 
        lv_bus=mv_bus, 
        sn_mva=100.0, 
        vn_hv_kv=vn_hv, 
        vn_lv_kv=6.6, 
        vk_percent=10.0, 
        vkr_percent=0.5, 
        pfe_kw=0.0, 
        i0_percent=0.0,
        name="BESS_StepUp_Transformer_6.6kV_to_HV"
    )
    print("Step-up transformer (6.6 kV / HV) successfully linked.")

    # 4. Connect the BESS Inverter / Storage Unit (modeled as a Static Generator) on the 6.6 kV side
    bess_active_power_mw = 50.0  # Controlled active power injection for thermal relief
    pp.create_sgen(
        net, 
        bus=mv_bus, 
        p_mw=bess_active_power_mw, 
        q_mvar=0.0, 
        name="BESS_100MW_Modular_Unit"
    )
    print(f"BESS active power injection of {bess_active_power_mw} MW initialized on the 6.6 kV side.")

    # 5. Run AC Power Flow validation
    pp.runpp(net)
    print("\n--- Power Flow Results Summary ---")
    print("Bus Voltages (p.u.):")
    print(net.res_bus[['vm_pu', 'va_degree']])
    
    print("\nTransformer Loading (%):")
    print(net.res_trafo[['loading_percent', 'pl_mw']])

if __name__ == "__main__":
    run_ieee9_bess_simulation()