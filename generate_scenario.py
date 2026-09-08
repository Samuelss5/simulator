from source.ScenarioGenerator import ScenarioGenerator

def main():

    import yaml


    file_path = 'networks_configurations.yaml'

    networks_configs = yaml.safe_load(open(file_path))

    # ______________________________
    # Networks Parameters
    # ______________________________

    pn = networks_configs['Fixed_service']
    sn = networks_configs['Dmimo_network']

    num_snapshots = networks_configs['num_snapshots']

    sys = ScenarioGenerator(pn, sn, num_snapshots)

    sys.run_networks_snapshots()

    sys.compute_inter_networks_links()


if __name__ == '__main__':
    main()

