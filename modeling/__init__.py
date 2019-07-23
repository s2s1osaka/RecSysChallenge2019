# for pandas's read_csv and to_csv function
target_dtype = {"gid": "str"
    , "impression": "str"
    , "price": "float16"
    , "position": "int8"
    , "n_imps": "int16"
    , "price_mean": "float16"
    , "price_std": "float16"
    , "user_id": "str"
    , "_session_id": "str"
    , "session_id": "str"
    , "timestamp": "str"
    , "timestamp_dt": "str"
    , "step": "int8"
    , "_step": "int8"
    , "reference": "str"
    , "platform": "str"
    , "city": "str"
    , "current_filters": "str"
    , "device": "str"
    , "country_name": "str"
    , "is_train": "int8"
    , "is_y": "int8"
    , "session_duration": "float16"
    , "step_duration": "float16"
    , "cf_sbp": "int8"
    , "cf_sbd": "int8"
    , "cf_sbr": "int8"
    , "cf_fod": "int8"
    , "cf_fsr": "int8"
    , "cf_bev": "int8"
    , "desktop": "int8"
    , "mobile": "int8"
    , "pos_rate": "float16"
    , "pos": "int8"
    , "price_norm": "float16"
    , "price_norm_rank": "float16"
    , "is_multiple_dist": "int8"
    , "co_pos_min": "float16"
    , "co_pos_max": "float16"
    , "co_pos_mean": "float16"
    , "co_pos_min_diff": "float16"
    , "co_pos_mean_diff": "float16"
    , "clickouted_pos_max_diff": "float16"
    , "impsocre": "float16"
    , "item_price_mean": "float16"
    , "discount_rate": "float16"
    , "item_id_x": "float16"
    , "pMicrowave": "int8"
    , "pRestaurant": "int8"
    , "pReception (24/7)": "int8"
    , "pFamily Friendly": "int8"
    , "pExcellent Rating": "int8"
    , "pAir Conditioning": "int8"
    , "pCable TV": "int8"
    , "pHotel": "int8"
    , "pConference Rooms": "int8"
    , "pFree WiFi (Rooms)": "int8"
    , "pTerrace (Hotel)": "int8"
    , "pHairdryer": "int8"
    , "pFrom 3 Stars": "int8"
    , "pSelf Catering": "int8"
    , "p2 Star": "int8"
    , "pLaundry Service": "int8"
    , "pElectric Kettle": "int8"
    , "p5 Star": "int8"
    , "pWashing Machine": "int8"
    , "pGood Rating": "int8"
    , "p4 Star": "int8"
    , "pVery Good Rating": "int8"
    , "pCot": "int8"
    , "pCar Park": "int8"
    , "p3 Star": "int8"
    , "pTelephone": "int8"
    , "pBusiness Hotel": "int8"
    , "pComputer with Internet": "int8"
    , "p1 Star": "int8"
    , "pSatisfactory Rating": "int8"
    , "pShower": "int8"
    , "pOpenable Windows": "int8"
    , "pNon-Smoking Rooms": "int8"
    , "pTelevision": "int8"
    , "pDesk": "int8"
    , "pWiFi (Public Areas)": "int8"
    , "pLuxury Hotel": "int8"
    , "pHouse / Apartment": "int8"
    , "pFree WiFi (Public Areas)": "int8"
    , "pBusiness Centre": "int8"
    , "pCentral Heating": "int8"
    , "pSatellite TV": "int8"
    , "pFridge": "int8"
    , "pWiFi (Rooms)": "int8"
    , "pFrom 2 Stars": "int8"
    , "pFree WiFi (Combined)": "int8"
    , "r6": "int8"
    , "r7": "int8"
    , "r8": "int8"
    , "r9": "int8"
    , "rating": "float16"
    , "r_price_mean": "float16"
    , "r_price_std": "float16"
    , "r_price_norm": "float16"
    , "star": "float16"
    , "s_price_mean": "float16"
    , "s_price_std": "float16"
    , "s_price_norm": "float16"
    , "item_id_y": "float16"
    , "prop_svd_1": "float16"
    , "prop_svd_2": "float16"
    , "prop_svd_3": "float16"
    , "prop_svd_4": "float16"
    , "prop_svd_5": "float16"
    , "prop_svd_6": "float16"
    , "prop_svd_7": "float16"
    , "prop_svd_8": "float16"
    , "prop_svd_9": "float16"
    , "prop_svd_10": "float16"
    , "pre_references": "str"
    , "precnt_ratio": "float16"
    , "interaction_item_rating_ratio": "float16"
    , "iif_ratio": "float16"
    , "iii_ratio": "float16"
    , "iid_ratio": "float16"
    , "sfi_ratio": "float16"
    , "co_ratio": "float16"
    , "cnt": "float16"
    , "last_reference": "str"
    , "last_timestamp": "str"
    , "is_last": "int8"
    , "elapsed_time_between_is_last": "float16"
    , "last_last_reference": "str"
    , "is_last_last": "int8"
    , "clickouted": "int8"
    , "sv2v_score": "float16"
    , "sv2v_score_mean": "float16"
    , "sv2v_score_std": "float16"
    , "sv2v_score_norm": "float16"
    , "clickouted_up2": "int8"
    , "clickouted_up1": "int8"
    , "clickouted_u1": "int8"
    , "clickouted_u2": "int8"
    , "clickouted_u3": "int8"
    , "clickouted_u4": "int8"
    , "clickouted_u5": "int8"
    , "is_last_up2": "int8"
    , "is_last_up1": "int8"
    , "is_last_u1": "int8"
    , "is_last_u2": "int8"
    , "is_last_u3": "int8"
    , "is_last_u4": "int8"
    , "is_last_u5": "int8"
    , "clickouted_sum": "int8"
    , "is_last_sum": "int8"
    , "ctrbycity": "float16"
    , "city_prob": "float16"
    , "country_name_prob": "float16"
    , "ctrbyplatform": "float16"
    , "platform_prob": "float16"
    , "ctrbyplatform_rank": "float16"
    , "it_count": "int8"
    , "is_zeroit": "int8"
    , "etbil_x_pr": "float16"
    , "lat_change of sort order": "float16"
    , "lat_clickout item": "float16"
    , "lat_filter selection": "float16"
    , "lat_interaction item deals": "float16"
    , "lat_interaction item image": "float16"
    , "lat_interaction item info": "float16"
    , "lat_interaction item rating": "float16"
    , "lat_search for destination": "float16"
    , "lat_search for item": "float16"
    , "lat_search for poi": "float16"
    , "rlr": "str"
    , "bayes_li": "float16"
    , "clicked": "int8"
    , "iired": "int8"
    , "iifed": "int8"
    , "iiied": "int8"
    , "iided": "int8"
    , "sfied": "int8"
    , "dropout_rate": "float16"
    , "all_dropout_rate": "float16"
    , "couted_price_mean": "float16"
    , "clickouted_price_diff": "float16"
    , "at_fa_coi": "float16"
    , "at_fa_coso": "float16"
    , "at_fa_fis": "float16"
    , "at_fa_iidea": "float16"
    , "at_fa_iii": "float16"
    , "at_fa_iiinfo": "float16"
    , "at_fa_iirat": "float16"
    , "at_fa_sfd": "float16"
    , "at_fa_sfi": "float16"
    , "at_fa_sfp": "float16"
    , "uniqueref_ratio": "float16"
    , "iir_cnt": "int8"
    , "iif_cnt": "int8"
    , "iii_cnt": "int8"
    , "iid_cnt": "int8"
    , "sfi_cnt": "int8"
    , "co_cnt": "int8"
    , "ctrbycity_rank": "float16"
    , "co_at_rate": "float16"
    , "iid_at_rate": "float16"
    , "iii_at_rate": "float16"
    , "iif_at_rate": "float16"
    , "iir_at_rate": "float16"
    , "sfi_at_rate": "float16"
    , "co_at_rate_rank": "float16"
    , "iid_at_rate_rank": "float16"
    , "iii_at_rate_rank": "float16"
    , "iif_at_rate_rank": "float16"
    , "iir_at_rate_rank": "float16"
    , "sfi_at_rate_rank": "float16"
    , "step_elapsed_mean": "float16"
    , "ref_elapsed_mean": "float16"
    , "co_ref_elapsed_mean": "float16"
    , "iid_ref_elapsed_mean": "float16"
    , "iii_ref_elapsed_mean": "float16"
    , "iif_ref_elapsed_mean": "float16"
    , "iir_ref_elapsed_mean": "float16"
    , "sfi_ref_elapsed_mean": "float16"
    , "elapsed_time": "float16"
    , "tsh24": "int8"
    , "ctrbytsh24": "float16"
    , "star_rating": "str"
    , "ctrbyprops": "float16"
    , "ctrbyitem": "float16"
    , "next_elapsed_time": "float16"
    , "next_elapsed_time_byco": "float16"
}
parse_dates = [11]

# for training
def get_id_cols():
    id_cols = ["gid", "impression"]
    return id_cols

def get_target_cols():
    target_cols = list(target_dtype.keys())
    target_cols.remove("gid")
    target_cols.remove("impression")
    target_cols.remove("user_id")
    target_cols.remove("_session_id")
    target_cols.remove("session_id")
    target_cols.remove("timestamp")
    target_cols.remove("timestamp_dt")
    target_cols.remove("reference")
    target_cols.remove("platform")
    target_cols.remove("city")
    target_cols.remove("current_filters")
    target_cols.remove("device")
    target_cols.remove("country_name")
    target_cols.remove("step")
    target_cols.remove("pre_references")
    target_cols.remove("last_reference")
    target_cols.remove("last_timestamp")
    target_cols.remove("last_last_reference")
    target_cols.remove("rlr")
    target_cols.remove("star_rating")
    target_cols.remove("is_train")
    target_cols.remove("is_y")
    target_cols.remove("pos")
    target_cols.remove("is_multiple_dist")
    target_cols.remove("item_id_x")
    target_cols.remove("item_id_y")
    target_cols.remove("price")
    target_cols.remove("position")
    target_cols.remove("n_imps")
    target_cols.remove("price_mean")
    target_cols.remove("price_std")
    target_cols.remove("item_price_mean")
    target_cols.remove("r_price_std")
    target_cols.remove("r_price_mean")
    target_cols.remove("s_price_mean")
    target_cols.remove("s_price_std")
    target_cols.remove("cnt")
    target_cols.remove("tsh24")
    target_cols.remove("r6")
    target_cols.remove("r7")
    target_cols.remove("r8")
    target_cols.remove("r9")
    target_cols.remove("clicked")
    return target_cols

# simple LightGBM parameters with simple higher parameters
lgbm_params = {'objective': 'lambdarank'
    , 'metric': 'ndcg'
    , 'ndcg_eval_at': {1, 3, 5}}

_lgbm_params = {'objective': 'lambdarank'
    , 'metric': 'ndcg'
    , 'ndcg_eval_at': {1, 3, 5}}

num_boost_round = 800

_num_boost_round = 10