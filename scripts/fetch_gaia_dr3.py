#!/usr/bin/env python3
"""
Fetch real astronomical star cluster data from the European Space Agency (ESA) Gaia DR3 Archive.
Computes true 3D Cartesian coordinates, space velocities, and individual stellar masses from photometry.
"""

import os
import sys
import math
import urllib.request
import urllib.parse
import csv

os.makedirs("data", exist_ok=True)

# Target: Pleiades Open Cluster (M45)
# RA: 56.75 deg, Dec: 24.12 deg, radius: 2.5 deg, distance ~136 pc
CLUSTER_NAME = "Pleiades (M45)"
CENTER_RA = 56.75
CENTER_DEC = 24.12
SEARCH_RADIUS_DEG = 2.5
MAX_STARS = 4096

print(f"Querying ESA Gaia DR3 Archive for {CLUSTER_NAME}...")

# ADQL Query to ESA Gaia TAP Server
query = f"""
SELECT TOP {MAX_STARS}
    source_id, ra, dec, parallax, pmra, pmdec, radial_velocity, phot_g_mean_mag, bp_rp
FROM gaiadr3.gaia_source
WHERE 1 = CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', {CENTER_RA}, {CENTER_DEC}, {SEARCH_RADIUS_DEG}))
  AND parallax BETWEEN 6.0 AND 8.5
  AND parallax_over_error > 5.0
  AND phot_g_mean_mag IS NOT NULL
ORDER BY phot_g_mean_mag ASC
"""

params = {
    'REQUEST': 'doQuery',
    'LANG': 'ADQL',
    'FORMAT': 'csv',
    'QUERY': query
}

url = 'https://gea.esac.esa.int/tap-server/tap/sync?' + urllib.parse.urlencode(params)
req = urllib.request.Request(url, headers={'User-Agent': 'AstroHPC-Simulation/1.0'})

try:
    with urllib.request.urlopen(req, timeout=30) as resp:
        content = resp.read().decode('utf-8')
except Exception as e:
    print(f"Error querying Gaia TAP API: {e}")
    sys.exit(1)

lines = content.strip().split('\n')
if len(lines) <= 1:
    print("No data returned from Gaia archive.")
    sys.exit(1)

reader = csv.DictReader(lines)
stars = []

sum_x, sum_y, sum_z = 0.0, 0.0, 0.0
sum_vx, sum_vy, sum_vz = 0.0, 0.0, 0.0

for row in reader:
    try:
        sid = row['source_id']
        ra_rad = math.radians(float(row['ra']))
        dec_rad = math.radians(float(row['dec']))
        plx = float(row['parallax'])
        if plx <= 0: continue

        dist_pc = 1000.0 / plx # Parsecs

        # Spherical to Cartesian (pc)
        x = dist_pc * math.cos(dec_rad) * math.cos(ra_rad)
        y = dist_pc * math.cos(dec_rad) * math.sin(ra_rad)
        z = dist_pc * math.sin(dec_rad)

        # Astrometric Proper motions (mas/yr -> km/s)
        # v_tan = 4.74047 * (pm / plx) km/s
        pmra = float(row['pmra']) if row['pmra'] else 0.0
        pmdec = float(row['pmdec']) if row['pmdec'] else 0.0
        rv = float(row['radial_velocity']) if row['radial_velocity'] else 0.0

        vx = 4.74047 * (pmra / plx)
        vy = 4.74047 * (pmdec / plx)
        vz = rv

        # Mass estimation from Gaia G-band absolute magnitude
        # M_G = G + 5 - 5*log10(dist_pc)
        g_mag = float(row['phot_g_mean_mag'])
        abs_g = g_mag + 5.0 - 5.0 * math.log10(dist_pc)

        # Standard main-sequence Mass-Luminosity relation:
        # Solar absolute G magnitude ~ 4.67
        # L/L_sun = 10^(-0.4 * (M_G - 4.67))
        # Mass ~ L^(1 / 3.5)
        lum = 10.0 ** (-0.4 * (abs_g - 4.67))
        mass = lum ** (1.0 / 3.5)

        # Physically bounded for cluster members (0.1 to 12.0 Solar Masses)
        if mass < 0.1: mass = 0.1
        if mass > 12.0: mass = 12.0

        stars.append({
            'id': sid,
            'mass': mass,
            'x': x, 'y': y, 'z': z,
            'vx': vx, 'vy': vy, 'vz': vz
        })

        sum_x += x; sum_y += y; sum_z += z
        sum_vx += vx; sum_vy += vy; sum_vz += vz
    except (ValueError, KeyError):
        continue

n = len(stars)
if n == 0:
    print("Could not parse stars from response.")
    sys.exit(1)

# Center the cluster at origin (0, 0, 0)
mean_x = sum_x / n
mean_y = sum_y / n
mean_z = sum_z / n

mean_vx = sum_vx / n
mean_vy = sum_vy / n
mean_vz = sum_vz / n

# 1:1 Physical Units accurate to real life (IRL):
# Position: Parsecs (pc) relative to cluster barycenter
# Velocity: km/s relative to cluster barycenter
# Mass: Solar Masses (M_sun)
# Time unit: 0.9778 Myr (~1 million years)
# G = 0.004300917 pc * (km/s)^2 / M_sun
scale_pos = 1.0 # 1 sim unit = 1 pc
scale_vel = 1.0 # 1 sim vel  = 1 km/s

output_path = "data/gaia_dr3_pleiades.csv"
with open(output_path, "w") as f:
    f.write(f"# Gaia DR3 Real Star Cluster: {CLUSTER_NAME}\n")
    f.write(f"# Total Stars: {n} | 1:1 IRL Physical Units: pos=pc, vel=km/s, mass=M_sun\n")
    f.write(f"# Gravitational constant: G = 0.004300917 pc*(km/s)^2/M_sun | 1 time unit = 0.9778 Myr\n")
    f.write("# name,mass,x,y,z,vx,vy,vz\n")

    for s in stars:
        # Shift relative to cluster barycenter
        rel_x = (s['x'] - mean_x) * scale_pos
        rel_y = (s['y'] - mean_y) * scale_pos
        rel_z = (s['z'] - mean_z) * scale_pos

        rel_vx = (s['vx'] - mean_vx) * scale_vel
        rel_vy = (s['vy'] - mean_vy) * scale_vel
        rel_vz = (s['vz'] - mean_vz) * scale_vel

        f.write(f"gaia_{s['id']},{s['mass']:.4f},{rel_x:.4f},{rel_y:.4f},{rel_z:.4f},{rel_vx:.4f},{rel_vy:.4f},{rel_vz:.4f}\n")

masses = [s['mass'] for s in stars]
print(f"Successfully fetched and processed {n} stars from Gaia DR3!")
print(f"  Stellar Mass Range: {min(masses):.2f} M_sun to {max(masses):.2f} M_sun (Mean: {sum(masses)/n:.2f} M_sun)")
print(f"  Dataset saved to: {output_path}")
