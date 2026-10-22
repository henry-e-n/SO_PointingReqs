#!/bin/bash
# Job name:
# SBATCH --job-name=lat_sso_sim
# Job output
#SBATCH --output slurm_outputs/%A_%a_lat_sso_sim.out
#SBATCH --error slurm_outputs/%A_%a_lat_sso_sim.err
# Email job changes
#SBATCH --mail-type=END
#SBATCH --mail-user=henry.nachman@utexas.edu
# Time limit (1 minute)
#SBATCH --time 04:00:00
# Number of MPI tasks needed for use case (example):
#SBATCH --ntasks=2
# Processors per task:
#SBATCH --cpus-per-task=16
# One task per node
#SBATCH --ntasks-per-node=4
# Set memory requirement
#SBATCH --mem 256GB

module use /shared_home/software/modulefiles/
module load soconda/stable

# Activate the custom environment
conda activate /shared_home/henachman/.conda/envs/toastdevenv
echo "Loaded custom conda environment"

sso_name="Deep56"

# Pick a tube and figure out the offset from boresight
tele="LAT"
tubes=i1 #(c1 i1 i3 i4 i5 i6 o6)
days=(1)
export OMP_NUM_THREADS=16
ntask=4
# export TOAST_LOGLEVEL=VERBOSE
source_size=2
scan_map="/cephfs/soukdata/user_data/henachman/sim_sky_maps/hr_cl_map_nosource.fits"
beam_dir="/shared_home/henachman/home/SO_PointingReqs/LAT_sims/"
current_date=$(date +%y%m%d_%H%M%S)
offset_enable="" # set to offset_
jitter_enable="" # set to jitter_ 
echo "---"
echo "CONFIGURATION:"
if [ -n "$offset_enable" ]; then
    echo "OFFSET: ENABLED"
    offset_tag="--pointing_offset.enable"
else
    echo "OFFSET: DISABLED"
    offset_tag="--pointing_offset.disable"
fi
if [ -n "$jitter_enable" ]; then
    echo "JITTER: ENABLED"
    jitter_tag="--pointing_jitter.enable"
else
    echo "JITTER: DISABLED"
    jitter_tag="--pointing_jitter.disable"
fi
echo "SCAN MAP: ${scan_map}"

for tube in "${tubes[@]}";
do
    case $tube in
        o6)
        bands=(f030 f040)
    	;;
        i1|i3|i4|i6)
        bands=(f090 f150)
    	;;
        c1|i5)
        bands=(f230 f290)
    	;;
        *)
    	echo "2 Unknown band: $band"
    	exit
    	;;
    esac
    # NOTE:  for wafer offsets, use the --wafer_slot option
    export -f conda
    offsets=$(PATH="/shared_home/henachman/.conda/envs/toastdevenv/bin:$PATH" get_wafer_offset --tube_slots $tube)
    WAFEROFFSET=$(echo "$offsets" | sed -n '2p')
    read -a WAFERARRAY <<< "$WAFEROFFSET"
    offset_az=${WAFERARRAY[0]}
    offset_el=${WAFERARRAY[1]}
    tube_radius=${WAFERARRAY[2]}
    
    for band_name in "${bands[@]}";
    do
        echo "tube ${tube} band ${band_name}"
        echo "offset_az = ${offset_az}"
        echo "offset_el = ${offset_el}"
        echo "tube_radius = ${tube_radius}"
        
        beam_file=${beam_dir}/${tele}_${band_name}_beam.h5
        
        for i in "${days[@]}";
        do
            # Simulate a scanning strategy
            t_start="2023-10-$i 00:00:00"
            t_stop="2023-10-$(($i+1)) 00:00:00"
            
            prefix=${sso_name}_${jitter_enable}${offset_enable}${band_name}_${current_date}
            # ${tube}
            prefix="${prefix// /_}"
            
            outdir="/cephfs/soukdata/user_data/henachman/sim_outputs/CMB/${prefix}"
            echo ${outdir}
            
            mkdir -p "${outdir}"
            schedule=${outdir}/schedule_${tube}_${sso_name}.txt
            `python config_writer.py "${outdir}" "scan_map=${scan_map}" "jitter_tag=${jitter_tag}" "sso_name=${sso_name}"`
            echo "---"
            toast_ground_schedule \
                --site-lat -22.958064 \
                --site-lon -67.786222 \
                --site-alt 5200 \
                --site-name ATACAMA \
                --telescope ${tele} \
                --patch-coord C \
                --elevations-deg "40,45,50,55,60" \
                --el-min-deg 40 \
                --el-max-deg 65 \
                --sun-el-max-deg 90 \
                --sun-avoidance-angle-deg 45 \
                --moon-avoidance-angle-deg 5 \
                --start "$t_start" \
                --stop "$t_stop" \
                --gap-s 86400 \
                --gap-small-s 0 \
                --ces-max-time-s 1200 \
                --fp-radius-deg 0 \
                --patch deep56,1.000,-6.0,6.0,38.0,-9.0 \
                --boresight-offset-az-deg ${offset_az} \
                --boresight-offset-el-deg ${offset_el} \
                --out "$schedule"

            # Run a simulation
            case $band_name in
                f030|f040)
            	export_key=tube_slot
            	fsample=200
            	;;
                f090|f150)
            	export_key=wafer_slot
            	fsample=200
            	;;
                f230|f290)
            	export_key=wafer_slot
            	fsample=200
            	;;
                *)
            	echo "2 Unknown band: $band"
            	exit
            	;;
            esac
            
            # NOTE:  These parameters can also be set with a config file...
            # The "names" of each operator in the command line arguments are just
            # the names given in the script being called (toast_so_sim in this
            # case).
            srun --mpi=pmi2 -n ${ntask} toast_so_sim \
                `#--corotate_lat.no_corotate_lat` \
                `# Do not make a map, just write the timestreams` \
                --mapmaker.disable \
                --filterbin.enable \
                --filterbin.no_write_cov \
                --filterbin.no_write_rcond \
                --filterbin.write_hdf5 \
                `# Instrument params` \
                --tube_slots ${tube} \
                --bands LAT_${band_name} \
                --sample_rate "${fsample}" \
                `# Observing schedule` \
                --schedule ${schedule} \
                `# Scanning params` \
                --sim_ground.enable \
                --sim_ground.turnaround_mask 2 \
                --sim_ground.scan_rate_az 'Quantity("1.5 deg / s")' \
                --sim_ground.scan_accel_az 'Quantity("3.0 deg / s2")' \
                `# Apparently this schedule command doesn't work ?--sim_ground.schedule_file ${schedule}` \
                `# Use fixed weather parameters` \
                `# --sim_ground.median_weather` \
                `# Apply Pointing` \
                --det_pointing_azel.quats "quats_azel" \
                --det_pointing_radec.quats "quats_radec" \
                --weights_azel.mode "IQU" \
                --weights_radec.mode "IQU" \
                `# Pixelization` \
                --pixels_healpix_radec.nside 8192 \
                `# Simulated sky signal from a input map` \
                --scan_map.enable \
                --scan_map.file ${scan_map} \
                --scan_map.save_map \
                --scan_map.save_pointing \
                `# Enable the sim pointing keys` \
                --det_pointing_radec_sim.enable \
                --det_pointing_azel_sim.enable \
                --scan_map_pixels.enable \
                --scan_map_pixels.nside 8192 \
                ${jitter_tag} \
                --pointing_jitter.sigma_az 60.0 \
                --pointing_jitter.sigma_el 60.0 \
                --pointing_jitter.sigma_psi 60.0 \
                ${offset_tag} \
                --pointing_offset.dxi 300 \
                --pointing_offset.deta 0 \
                `# Simulated SSO` \
                `# --sim_sso.enable` \
                `# --sim_sso.sso_name ${sso_name}` \
                `# --sim_sso.beam_file ${beam_file}` \
                `# Simulated atmosphere params (high resolution)` \
                `# --sim_atmosphere.enable` \
                `# --sim_atmosphere.field_of_view 'Quantity("6 deg")'` \
                `# --sim_atmosphere.cache_dir '/cephfs/soukdata/user_data/henachman/sim_outputs/toast_cache'` \
                `# Simulated atmosphere params (coarse resolution)` \
                `# --sim_atmosphere_coarse.enable` \
                `# --sim_atmosphere_coarse.field_of_view 'Quantity("6 deg")'` \
                `# Noise simulation (from elevation-modulated focalplane parameters)` \
                --sim_noise.enable \
                `# Gain mismatch` \
                `#--gainscrambler.enable` \
                `#--gainscrambler.sigma 0.01` \
                `# Timeconstant convolution` \
                `#--convolve_time_constant.disable` \
                `#--convolve_time_constant.tau '3 ms'` \
                `# Write to HDF5` \
                --save_hdf5.enable \
                --out_dir ${outdir} \
                --job_group_size ${ntask} | tee "${outdir}/log"
        done
    done
done

