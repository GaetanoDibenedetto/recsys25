def get_cp_score(gender, age):
    if gender == "M":
        if 20 <= age <= 45:
            return 25
        else:
            return 20
    elif gender == "W":
        if 20 <= age <= 45:
            return 20
        else:
            return 15

def get_vm_score(vertical_distance):
    vm = 0
    if vertical_distance > 175:
        vm = 0
    elif vertical_distance == 170:
        vm = 0.70
    elif vertical_distance > 160:
        vm = 0.75
    elif vertical_distance > 150:
        vm = 0.78
    elif vertical_distance > 140:
        vm = 0.81
    elif vertical_distance > 130:
        vm = 0.84
    elif vertical_distance > 120:
        vm = 0.87
    elif vertical_distance > 110:
        vm = 0.90
    elif vertical_distance > 100:
        vm = 0.93
    elif vertical_distance > 90:
        vm = 0.96
    elif vertical_distance > 80:
        vm = 0.99
    elif vertical_distance > 75:
        vm = 1.00
    elif vertical_distance > 70:
        vm = 0.99
    elif vertical_distance > 60:
        vm = 0.96
    elif vertical_distance > 50:
        vm = 0.93
    elif vertical_distance > 40:
        vm = 0.90
    elif vertical_distance > 30:
        vm = 0.87
    elif vertical_distance > 20:
        vm = 0.84
    elif vertical_distance > 10:
        vm = 0.81
    elif vertical_distance >= 0:
        vm = 0.78
    return vm

def get_dm_score(vertical_distance):
    dm = 0 
    if vertical_distance > 175:
        dm = 0
    elif vertical_distance == 175:
        dm = 0.85
    elif vertical_distance > 160:
        dm = 0.85
    elif vertical_distance > 145:
        dm = 0.85
    elif vertical_distance > 130:
        dm = 0.86
    elif vertical_distance > 115:
        dm = 0.86
    elif vertical_distance > 100:
        dm = 0.87
    elif vertical_distance > 85:
        dm = 0.87
    elif vertical_distance > 70:
        dm = 0.88
    elif vertical_distance > 55:
        dm = 0.90
    elif vertical_distance > 40:
        dm = 0.93
    elif vertical_distance > 25: # da controllare
        dm = 0.93
    elif vertical_distance <= 25:
        dm = 1
    return dm

def get_hm_score(horizontal_distance):
    hm = 0
    if horizontal_distance > 63:
        hm = 0
    elif horizontal_distance == 63:
        hm = 0.40
    elif horizontal_distance > 60:
        hm = 0.42
    elif horizontal_distance > 58:
        hm = 0.43
    elif horizontal_distance > 56:
        hm = 0.45
    elif horizontal_distance > 54:
        hm = 0.46
    elif horizontal_distance > 52:
        hm = 0.48
    elif horizontal_distance > 50:
        hm = 0.50
    elif horizontal_distance > 48:
        hm = 0.52
    elif horizontal_distance > 46:
        hm = 0.54
    elif horizontal_distance > 44:
        hm = 0.57
    elif horizontal_distance > 42:
        hm = 0.60
    elif horizontal_distance > 40:
        hm = 0.63
    elif horizontal_distance > 38:
        hm = 0.66
    elif horizontal_distance > 36:
        hm = 0.69
    elif horizontal_distance > 34:
        hm = 0.74
    elif horizontal_distance > 32:
        hm = 0.78
    elif horizontal_distance > 30:
        hm = 0.83
    elif horizontal_distance > 28:
        hm = 0.89
    elif horizontal_distance > 25: #da controllare
        hm = 0.89
    elif horizontal_distance <= 25:
        hm = 1
    return hm

def get_am_score(degree):
    am = 0
    if degree > 135:
        am = 0
    elif degree == 135:
        am = 0.57
    elif degree > 105:
        am = 0.66
    elif degree > 75:
        am = 0.76
    elif degree > 60:
        am = 0.81
    elif degree > 45:
        am = 0.86
    elif degree > 30:
        am = 0.90
    elif degree > 15:
        am = 0.95
    elif degree > 0: # da controllare
        am = 0.95
    elif degree == 0:
        am = 1
    return am

def get_cm_score(judgment):
    if judgment == "good":
        return 1
    elif judgment == "intermediate":
        return 0.95
    elif judgment == "bad":
        return 0.90

def get_fm_score(frequency):
    # TODO
    return 1

def get_etm_score(mmc):
    # TODO
    return 1

def get_om_score(om):
    # if the lift is only with one hand
    if om == True:
        return 0.6
    else:
        return 1

def get_pm_score(pm):
    # if the lift is done by two operators
    if pm:
        return 0.85
    else:
        return 1


def compute_reccomended_weight(
    gender="M",
    age=25,
    min_hand_floor_distance_vertical=None,
    vertical_lifting_distance=None,
    max_orizontal_hands_mid_ankle_distance=None,
    torso_torsion=0,
    judgment="intermediate",
    lifting_frequency=1,
    etm=1,
    one_limb_lifting=False,
    two_operators_lifting=False,
):
    """
    This function computes the recommended weight for a person based on the inputs provided.

    Args:
        gender (str, optional): M for man or W for woman. Defaults to "M".
        age (int, optional): age of the subject lifting the package. Defaults to 25.
        min_hand_floor_distance_vertical (_type_, optional): minimum vertical distance from hands to floor. Defaults to None.
        vertical_lifting_distance (_type_, optional): vertical distance of lifting. Defaults to None.
        max_orizontal_hands_mid_ankle_distance (_type_, optional): maximum horizontal distance between hands and mid hankles. Defaults to None.
        torso_torsion (int, optional): torsion of torso (in degrees). Defaults to 0.
        judgment (str, optional): subjective judgment for the lifting, could it be "good", "intermediate" or "bad". Defaults to "intermediate".
        lifting_frequency (int, optional): lifting frequency (number of lifts per minute) in relation to duration. Defaults to 1.
        etm (int, optional): multiplier for MMC times over 480 min. Defaults to 1.
        one_limb_lifting (bool, optional): lifts with only one limb. Defaults to False.
        two_operators_lifting (bool, optional): lifted by two operators. Defaults to False.

    Returns:
        double: value in kg of the recommended weight
    """

    assert gender in ["M", "W"]
    assert age > 0
    assert min_hand_floor_distance_vertical is not None
    assert vertical_lifting_distance is not None
    assert max_orizontal_hands_mid_ankle_distance is not None
    assert torso_torsion >= 0
    assert judgment in ["good", "intermediate", "bad"]
    assert lifting_frequency > 0
    assert etm > 0
    assert isinstance(one_limb_lifting, bool)
    assert isinstance(two_operators_lifting, bool)

    lc = get_cp_score(gender, age)
    vm = 0 if min_hand_floor_distance_vertical > 175 else 1 - (0.003 * abs(min_hand_floor_distance_vertical - 75))
    dm = 1 if vertical_lifting_distance <= 25 else (0 if vertical_lifting_distance > 175 else 0.82 + (4.5/vertical_lifting_distance))
    hm = 1 if max_orizontal_hands_mid_ankle_distance <= 25 else (0 if max_orizontal_hands_mid_ankle_distance > 63 else 25/max_orizontal_hands_mid_ankle_distance)
    am = 0 if torso_torsion > 135 else 1 - (0.0032 * torso_torsion)
    cm = get_cm_score(judgment)
    fm = get_fm_score(lifting_frequency)
    etm = get_etm_score(etm)
    om = get_om_score(one_limb_lifting)
    pm = get_pm_score(two_operators_lifting)

    recommended_weight = lc * vm * dm * hm * am * cm * fm * etm * om * pm
    
    # print(round(lc, 2), round(vm, 2), round(dm, 2), round(hm, 2), round(am, 2), round(cm, 2), round(fm, 2), round(etm, 2), round(om, 2), round(pm, 2))
    return recommended_weight


def compute_reccomended_weight_simplified(
    gender="M",
    age=25,
    min_hand_floor_distance_vertical=None,
    vertical_lifting_distance=None,
    max_orizontal_hands_mid_ankle_distance=None,
    torso_torsion=0,
    judgment="intermediate",
    lifting_frequency=1,
    etm=1,
    one_limb_lifting=False,
    two_operators_lifting=False,
):
    """
    This function computes the recommended weight for a person based on the inputs provided.

    Args:
        gender (str, optional): M for man or W for woman. Defaults to "M".
        age (int, optional): age of the subject lifting the package. Defaults to 25.
        min_hand_floor_distance_vertical (_type_, optional): minimum vertical distance from hands to floor. Defaults to None.
        vertical_lifting_distance (_type_, optional): vertical distance of lifting. Defaults to None.
        max_orizontal_hands_mid_ankle_distance (_type_, optional): maximum horizontal distance between hands and mid hankles. Defaults to None.
        torso_torsion (int, optional): torsion of torso (in degrees). Defaults to 0.
        judgment (str, optional): subjective judgment for the lifting, could it be "good", "intermediate" or "bad". Defaults to "intermediate".
        lifting_frequency (int, optional): lifting frequency (number of lifts per minute) in relation to duration. Defaults to 1.
        etm (int, optional): multiplier for MMC times over 480 min. Defaults to 1.
        one_limb_lifting (bool, optional): lifts with only one limb. Defaults to False.
        two_operators_lifting (bool, optional): lifted by two operators. Defaults to False.

    Returns:
        double: value in kg of the recommended weight
    """

    assert gender in ["M", "W"]
    assert age > 0
    assert min_hand_floor_distance_vertical is not None
    assert vertical_lifting_distance is not None
    assert max_orizontal_hands_mid_ankle_distance is not None
    assert torso_torsion >= 0
    assert judgment in ["good", "intermediate", "bad"]
    assert lifting_frequency > 0
    assert etm > 0
    assert isinstance(one_limb_lifting, bool)
    assert isinstance(two_operators_lifting, bool)

    cp = get_cp_score(gender, age)
    vm = get_vm_score(min_hand_floor_distance_vertical)
    dm = get_dm_score(vertical_lifting_distance)
    hm = get_hm_score(max_orizontal_hands_mid_ankle_distance)
    am = get_am_score(torso_torsion)
    cm = get_cm_score(judgment)
    fm = get_fm_score(lifting_frequency)
    etm = get_etm_score(etm)
    om = get_om_score(one_limb_lifting)
    pm = get_pm_score(two_operators_lifting)

    recommended_weight = cp * vm * dm * hm * am * cm * fm * etm * om * pm

    print(round(cp, 2), round(vm, 2), round(dm, 2), round(hm, 2), round(am, 2), round(cm, 2), round(fm, 2), round(etm, 2), round(om, 2), round(pm, 2))

    return recommended_weight


if __name__ == "__main__":
    import random
    set_seed = 9999
    random.seed(set_seed)
    min_hand_floor_distance_vertical = random.randint(0, 176)
    vertical_lifting_distance = random.randint(0, 176)
    max_orizontal_hands_mid_ankle_distance = random.randint(0, 64)
    torso_torsion = 0
    judgment = "intermediate"
    lifting_frequency = 1
    etm=1
    one_limb_lifting=False
    two_operators_lifting=False

    test = compute_reccomended_weight(
        gender="M",
        age=25,
        min_hand_floor_distance_vertical=min_hand_floor_distance_vertical,
        vertical_lifting_distance=vertical_lifting_distance,
        max_orizontal_hands_mid_ankle_distance=max_orizontal_hands_mid_ankle_distance,
        torso_torsion=torso_torsion,
        judgment=judgment,
        lifting_frequency=lifting_frequency,
        etm=etm,
        one_limb_lifting=one_limb_lifting,
        two_operators_lifting=two_operators_lifting,
    )

    test2 = compute_reccomended_weight_simplified(
        gender="M",
        age=25,
        min_hand_floor_distance_vertical=min_hand_floor_distance_vertical,
        vertical_lifting_distance=vertical_lifting_distance,
        max_orizontal_hands_mid_ankle_distance=max_orizontal_hands_mid_ankle_distance,
        torso_torsion=torso_torsion,
        judgment=judgment,
        lifting_frequency=lifting_frequency,
        etm=etm,
        one_limb_lifting=one_limb_lifting,
        two_operators_lifting=two_operators_lifting,
    )

    print(test)
    print(test2)
