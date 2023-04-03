# Pre-Processing the Original Dataset

### 1. Download the data
<table><tbody>
<th valign="bottom"></th>
<th valign="bottom">train</th>
<th valign="bottom">validation</th>
<th valign="bottom">test</th>
<tr><td align="left">1 Mpx</td>
<td align="center"><a href="https://download.ifi.uzh.ch/rpg/RVT/datasets/gen4_tar/train.tar">download</a></td>
<td align="center"><a href="https://download.ifi.uzh.ch/rpg/RVT/datasets/gen4_tar/val.tar">download</a></td>
<td align="center"><a href="https://download.ifi.uzh.ch/rpg/RVT/datasets/gen4_tar/test.tar">download</a></td>
</tr>
<tr><td align="left">crc32</td>
<td align="center"><tt>d677488a</tt></td>
<td align="center"><tt>72f13c3e</tt></td>
<td align="center"><tt>643e61ef</tt></td>
</tr>
<tr><td align="left">Gen1</td>
<td align="center"><a href="https://download.ifi.uzh.ch/rpg/RVT/datasets/gen1_tar/train.tar">download</a></td>
<td align="center"><a href="https://download.ifi.uzh.ch/rpg/RVT/datasets/gen1_tar/val.tar">download</a></td>
<td align="center"><a href="https://download.ifi.uzh.ch/rpg/RVT/datasets/gen1_tar/test.tar">download</a></td>
</tr>
<tr><td align="left">crc32</td>
<td align="center"><tt>3d23bd30</tt></td>
<td align="center"><tt>cc802022</tt></td>
<td align="center"><tt>cdd4fd69</tt></td>
</tr>
</tbody></table>

### 2. Extract the tar files
The following directory structure is assumed:

```
data_dir
├── test
│   ├── ..._bbox.npy
│   ├── ..._td.dat.h5
│   ...
│
├── train
│   ├── ....npy
│   ├── ..._td.dat.h5
│   ...
│
└── val
    ├── ..._bbox.npy
    ├── ..._td.dat.h5
    ... 
```

### 3. Run the pre-processing script
`${DATA_DIR}` should point to the directory structure mentioned above.
`${DEST_DIR}` should point to the directory to which the data will be written.

For the 1 Mpx dataset:
```Bash
NUM_PROCESSES=20  # set to the number of parallel processes to use
python preprocess_dataset.py ${DATA_DIR} ${DEST_DIR} conf_preprocess/representation/stacked_hist.yaml \
conf_preprocess/extraction/const_duration.yaml conf_preprocess/filter_gen4.yaml -ds gen4 -np ${NUM_PROCESSES}
```

For the Gen1 dataset:
```Bash
NUM_PROCESSES=20  # set to the number of parallel processes to use
python preprocess_dataset.py ${DATA_DIR} ${DEST_DIR} conf_preprocess/representation/stacked_hist.yaml \
conf_preprocess/extraction/const_duration.yaml conf_preprocess/filter_gen1.yaml -ds gen1 -np ${NUM_PROCESSES}
```
