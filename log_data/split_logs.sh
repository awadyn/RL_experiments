#!/bin/bash

# isolate logs with rapl = 135
echo "LOGS WITH RAPL = 135"
echo "EBBRT.."
cat ebbrt_features_28.csv | while read line; do rapl=$(echo $line | cut -d ' ' -f5); if [ $rapl == 135 ]; then echo $line >> ebbrt_features_28_rapl_135.csv; fi; done
echo "LINUX.."
cat linux_features_28.csv | while read line; do rapl=$(echo $line | cut -d ' ' -f5); if [ $rapl == 135 ]; then echo $line >> linux_features_28_rapl_135.csv; fi; done


echo "LOGS WITH RAPL = 135 + ITR-DELAY = X"
# isolate logs with different itr-delay values
echo "EBBRT.."
ebbrt_itr_delays=$(cat ebbrt_features_28_rapl_135.csv | cut -d ' ' -f3 | sort -n | uniq)
# creating feature files for each itr-delay value
for i in $(echo $ebbrt_itr_delays); do echo $i; cat ebbrt_features_28_rapl_135.csv | while read line; do itr=$(echo $line | cut -d ' ' -f3); if [ $itr == $i ]; then echo $line >> ebbrt_features_28_rapl_135_itr_$i.csv; fi; done; done
echo "LINUX.."
linux_itr_delays=$(cat linux_features_28_rapl_135.csv | cut -d ' ' -f3 | sort -n | uniq)
for i in $(echo $linux_itr_delays); do echo $i; cat linux_features_28_rapl_135.csv | while read line; do itr=$(echo $line | cut -d ' ' -f3); if [ $itr == $i ]; then echo $line >> linux_features_28_rapl_135_itr_$i.csv; fi; done; done


echo "LOGS WITH RAPL = 135 + QPS = X"
# similarly isolate per qps
echo "EBBRT.."
ebbrt_qpses=$(cat ebbrt_features_28_rapl_135.csv | cut -d ' ' -f13 | sort -n | uniq)
for q in $(echo $ebbrt_qpses); do echo $q; cat ebbrt_features_28_rapl_135.csv | while read line; do qps=$(echo $line | cut -d ' ' -f13); if [ $qps == $q ]; then echo $line >> ebbrt_features_28_rapl_135_qps_$q.csv; fi; done; done
echo "LINUX.."
linux_qpses=$(cat linux_features_28_rapl_135.csv | cut -d ' ' -f13 | sort -n | uniq)
for q in $(echo $linux_qpses); do echo $q; cat linux_features_28_rapl_135.csv | while read line; do qps=$(echo $line | cut -d ' ' -f13); if [ $qps == $q ]; then echo $line >> linux_features_28_rapl_135_qps_$q.csv; fi; done; done


echo "LOGS WITH RAPL = 135 + DVFS = X"
echo "EBBRT.."
ebbrt_dvfses=$(cat ebbrt_features_28_rapl_135.csv | cut -d ' ' -f4 | sort -n | uniq)
for d in $(echo $ebbrt_dvfses); do echo $d; cat ebbrt_features_28_rapl_135.csv | while read line; do dvfs=$(echo $line | cut -d ' ' -f4); if [ $dvfs == $d ]; then echo $line >> ebbrt_features_28_rapl_135_dvfs_$d.csv; fi; done; done
echo "LINUX.."
linux_dvfses=$(cat linux_features_28_rapl_135.csv | cut -d ' ' -f4 | sort -n | uniq)
for d in $(echo $linux_dvfses); do echo $d; cat linux_features_28_rapl_135.csv | while read line; do dvfs=$(echo $line | cut -d ' ' -f4); if [ $dvfs == $d ]; then echo $line >> linux_features_28_rapl_135_dvfs_$d.csv; fi; done; done


echo "LOGS WITH RAPL = 135 + DVFS = X + ITR-DELAY = Y"
echo "EBBRT.."
for d in $(echo $ebbrt_dvfses); do echo $d; cat ebbrt_features_28_rapl_135.csv | while read line; do dvfs=$(echo $line | cut -d ' ' -f4); itr=$(echo $line | cut -d ' ' -f3); if [ $dvfs == $d ]; then for i in $(echo $ebbrt_itr_delays); do if [ $itr == $i ]; then echo $line >> ebbrt_features_28_rapl_135_dvfs_$d\_itr_$i.csv; fi; done; fi; done; done
echo "LINUX.."
for d in $(echo $linux_dvfses); do echo $d; cat linux_features_28_rapl_135.csv | while read line; do dvfs=$(echo $line | cut -d ' ' -f4); itr=$(echo $line | cut -d ' ' -f3); if [ $dvfs == $d ]; then for i in $(echo $linux_itr_delays); do if [ $itr == $i ]; then echo $line >> linux_features_28_rapl_135_dvfs_$d\_itr_$i.csv; fi; done; fi; done; done


echo "LOGS WITH RAPL = 135 + QPS = X + ITR-DELAY = Y"
echo "EBBRT.."
for q in $(echo $ebbrt_qpses); do echo $q; cat ebbrt_features_28_rapl_135.csv | while read line; do qps=$(echo $line | cut -d ' ' -f13); itr=$(echo $line | cut -d ' ' -f3); if [ $qps == $q ]; then for i in $(echo $ebbrt_itr_delays); do if [ $itr == $i ]; then echo $line >> ebbrt_features_28_rapl_135_qps_$q\_itr_$i.csv; fi; done; fi; done; done
echo "LINUX.."
for q in $(echo $linux_qpses); do echo $q; cat linux_features_28_rapl_135.csv | while read line; do qps=$(echo $line | cut -d ' ' -f13); itr=$(echo $line | cut -d ' ' -f3); if [ $qps == $q ]; then for i in $(echo $linux_itr_delays); do if [ $itr == $i ]; then echo $line >> linux_features_28_rapl_135_qps_$q\_itr_$i.csv; fi; done; fi; done; done


echo "LOGS WITH RAPL = 135 + QPS = X + DVFS = Y"
echo "EBBRT.."
for q in $(echo $ebbrt_qpses); do echo $q; cat ebbrt_features_28_rapl_135.csv | while read line; do qps=$(echo $line | cut -d ' ' -f13); dvfs=$(echo $line | cut -d ' ' -f4); if [ $qps == $q ]; then for d in $(echo $ebbrt_dvfses); do if [ $dvfs == $d ]; then echo $line >> ebbrt_features_28_rapl_135_qps_$q\_dvfs_$d.csv; fi; done; fi; done; done
echo "LINUX.."
for q in $(echo $linux_qpses); do echo $q; cat linux_features_28_rapl_135.csv | while read line; do qps=$(echo $line | cut -d ' ' -f13); dvfs=$(echo $line | cut -d ' ' -f4); if [ $qps == $q ]; then for d in $(echo $linux_dvfses); do if [ $dvfs == $d ]; then echo $line >> linux_features_28_rapl_135_qps_$q\_dvfs_$d.csv; fi; done; fi; done; done




