from sklearn.model_selection import train_test_split

def split_data(t_set, f_set, set_validation_size=0.15, set_test_size=0.1):
    #Split data into training, validation and test sets using stratification.
    #Training size will always be the remainding percentage of the set.
    
    fTrn, fTst, tTrn, tTst = train_test_split(
        f_set, t_set,
        test_size=set_test_size,
        random_state=42,
        stratify=t_set
    )

    fTrn, fVld, tTrn, tVld = train_test_split(
        fTrn, tTrn,
        test_size=set_validation_size/(1-set_test_size),
        random_state=42,
        stratify=tTrn
    )
    return fTrn, tTrn, fVld, tVld, fTst, tTst