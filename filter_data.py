def filter_train(df, filter_ct, drop_dupes=False):
    # Filter rows where the sum of votes across all label columns is greater than filter_ct
    label_columns = ['seizure_vote', 'lpd_vote', 'gpd_vote', 'lrda_vote', 'grda_vote', 'other_vote']
    df = df[df[label_columns].sum(axis=1) > filter_ct].copy()

    # Create a unique identifier combining eeg_id and label columns
    df['eeg_id_l'] = df[['eeg_id'] + label_columns].astype(str).agg('_'.join, axis=1)

    rows = []
    # Remove overlapping and redundant rows
    for eeg_id_l in tqdm(df['eeg_id_l'].unique(), desc="Processing Groups"):
        df0 = df[df['eeg_id_l'] == eeg_id_l].reset_index(drop=True).copy()
        offsets = df0['spectrogram_label_offset_seconds'].astype(int).values
        x = np.zeros(offsets.max() + 600)
        for o in offsets:
            x[o:o + 600] += 1
        best_idx = np.argmax([x[o:o + 600].sum() for o in offsets])
        rows.append(df0.iloc[best_idx])

    filtered_df = pd.DataFrame(rows)

    # Drop duplicates
    if drop_dupes:
        filtered_df = filtered_df.drop_duplicates(subset='eeg_id').copy()

    return filtered_df
# Validation set, no dupes, high quality data
df_filtered_golden = filter_train(df.copy(), filter_ct=8, drop_dupes=True)

# Regular training set
df_filtered_regular = filter_train(df.copy(), filter_ct=8, drop_dupes=False)

# Large training set with low count votes
df_filtered_large = filter_train(df.copy(), filter_ct=0, drop_dupes=False)

df_filtered_golden.to_csv('filtered_data_golden.csv', index=False)
df_filtered_regular.to_csv('filtered_data_regular.csv', index=False)
df_filtered_large.to_csv('filtered_data_large.csv', index=False)