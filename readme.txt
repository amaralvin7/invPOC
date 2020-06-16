Contains code and supporting files to perform inversion of POC data.

 - invP_v**.py: script to perform inversion
 - poc_all.xlsk: contains all POC data, used for POC/cp relationship. Formerly poc_v2_mmol.csv
 - poc_means.xlsx: contains mean POC data, used for data inversion. Formerly POC_data_forOI_v2.xlsx
 - scrpdata.mat: matlab file containing cp metadata from the Sally Ride, used for POC/cp relationship. Need for filtering out bad casts.
 - cp_bycast.mat: a single matlab matrix that contains cp profiles from casts corresponding to the column number
 - castmatch_v2.csv: matching CTD and pump casts on the Sally Ride