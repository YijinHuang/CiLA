template = 'this is a fundus image '
def add_template(labels):
    return [template + label for label in labels]

lesion_type = [
    'without lesion',
    'with pale yellow or white lesions with ill-defined edges',
    'with small white or yellowish lesions with sharp margins',
    'with small red dots lesions',
    'with dense, dark red and sharply outlined lesions'
]
eyeside = [
    'without optic disc',
    'with optic disc on the left',
    'with optic disc on the right',
]
peripapillary = [
    'with lesions around the optic disc',
    'without lesion around the optic disc'
]
lesion_distribution = [
    'without lesion',
    'with scattered lesions',
    'with clustered lesions',
    'with isolated lesions',
    'with patchy lesions'
]
lesion_size = [
    'without lesion',
    'with tiny lesions',
    'with small lesions',
    'with medium-sized lesions',
    'with large lesions'
]
lesion_amount = [
    'without lesion',
    'with a few lesions',
    'with several lesions',
    'with numerous lesions'
]
background = [
    'with orange background',
    'with pink background',
    'with brown background',
    'with dark red background'
]
optic_disc_color = [
    'without optic disc',
    'with orange optic disc',
    'with pale optic disc',
    'with blanched optic disc'
]
optic_disc_border = [
    'without optic disc',
    'with optic disc with ill-defined borders',
    'with optic disc with well-defined borders'
]
optic_disc_shape = [
    'without optic disc',
    'with circular optic disc',
    'with circular oval disc',
    'with circular symmetrical disc',
    'with circular asymmetry disc'
]
optic_disc_size = [
    'without optic disc',
    'with small optic disc',
    'with average-sized optic disc',
    'with large optic disc'
]
vessel_color = [
    'without vessels',
    'with light red vessels',
    'with dark red vessels'
]
vessel_width = [
    'without vessels',
    'with narrow vessels',
    'with wide vessels'
]
vessel_clarity = [
    'without vessels',
    'with clear and distinct vessels',
    'with blur and fuzzy vessels'
]
clarity = [
    'this is a clear fundus image',
    'this is a blur fundus image',
    'this is a broken fundus image'
]
diabetic_retinopathy = [
    'without referable lesion',
    'with few microaneurysms',
    'with retinal haemorrhages in few quadrants',
    'with severe haemorrhages in all four quadrants',
    'with neovascularization at the optic disc'
]
diabetic_macular_edema = [
    'without exudate around the macula center',
    'with exudates around the macula center'
]
age_related_macular_degeneration = [
    'without drusen',
    'with drusen'
]
dr_grading_categories = [
    'of a healthy eyes',
    'of mild diabetic retinopathy',
    'of moderate diabetic retinopathy',
    'of severe diabetic retinopathy',
    'of proliferative diabetic retinopathy',
]
dr_grading_types = [
    'with no visible lesions',
    'with a few small microaneurysms',
    'with several microaneurysms and mild hemorrhages',
    'with numerous microaneurysms and hemorrhages',
    'with neovascularization'
]
dr_grading_details = [
    'with no visible lesions',
    'with a few small red dot-like spots',
    'with several small red dots and scattered dark red spots',
    'with numerous red dots and larger, dark blot-like spots',
    'with abnormal thin, fragile blood vessels'
]
dr_detection_categories = [
    'of a healthy eyes',
    'of diabetic retinopathy'
]
dr_detection_types = [
    'with no visible lesions',
    'with microaneurysms, haemorrhages or neovascularization',
]
dr_detection_details = [
    'with no visible lesions',
    'with small red dot-like spots, dark red blot-like spots or abnormal thin, fragile blood vessels',
]

def merge(*labels):
    collections = labels[0].copy()
    for label in labels[1:]:
        collections.update(label)
    return collections

lesion_labels = {
    'lesion_type': add_template(lesion_type),
    'lesion_distribution': add_template(lesion_distribution),
    'peripapillary': add_template(peripapillary),
    'lesion_amount': add_template(lesion_amount),
    'lesion_size': add_template(lesion_size),
}

optic_disc_labels = {
    'eyeside': add_template(eyeside),
    'peripapillary': add_template(peripapillary),
    'optic_disc_color': add_template(optic_disc_color),
    'optic_disc_border': add_template(optic_disc_border),
    'optic_disc_shape': add_template(optic_disc_shape),
    'optic_disc_size': add_template(optic_disc_size),
}

vessel_labels = {
    'vessel_color': add_template(vessel_color),
    'vessel_width': add_template(vessel_width),
    'vessel_clarity': add_template(vessel_clarity),
}

disease_labels = {
    'diabetic_retinopathy': add_template(diabetic_retinopathy),
    'diabetic_macular_edema': add_template(diabetic_macular_edema),
    'age_related_macular_degeneration': add_template(age_related_macular_degeneration),
}

dr_grading_labels = {
    'dr_grading_categories': add_template(dr_grading_categories),
    'dr_grading_types': add_template(dr_grading_types),
    'dr_grading_details': add_template(dr_grading_details),
}

dr_detection_labels = {
    'dr_detection_categories': add_template(dr_detection_categories),
    'dr_detection_types': add_template(dr_detection_types),
    'dr_detection_details': add_template(dr_detection_details),
}

dr_labels = merge(dr_grading_labels, dr_detection_labels)

other_labels = {
    'background': add_template(background),
    'clarity': clarity,
}

dr_all = {
    'lesion_type': add_template(lesion_type),
    'lesion_distribution': add_template(lesion_distribution),
    'lesion_amount': add_template(lesion_amount),
    'lesion_size': add_template(lesion_size),
    'dr_grading_categories': add_template(dr_grading_categories),
    'dr_grading_types': add_template(dr_grading_types),
    'dr_grading_details': add_template(dr_grading_details)
}

dr_lesions = {
    'lesion_type': add_template(lesion_type),
    'lesion_distribution': add_template(lesion_distribution),
    'lesion_amount': add_template(lesion_amount),
    'lesion_size': add_template(lesion_size)
}

lesion_type = {
    'lesion_type': add_template(lesion_type)
}

lesion_distribution = {
    'lesion_distribution': add_template(lesion_distribution)
}

lesion_amount = {
    'lesion_amount': add_template(lesion_amount)
}

lesion_size = {
    'lesion_size': add_template(lesion_size)
}

dr_labels = {
    'dr_grading_categories': add_template(dr_grading_categories),
    'dr_grading_types': add_template(dr_grading_types),
    'dr_grading_details': add_template(dr_grading_details)    
}

label_collections = {
    'all': merge(lesion_labels, optic_disc_labels, vessel_labels, disease_labels, dr_labels, other_labels),
    'lesion': lesion_labels,
    'optic_disc': optic_disc_labels,
    'vessel': vessel_labels,
    'disease': disease_labels,
    'dr_grading': dr_grading_labels,
    'dr_detection': dr_detection_labels,
    'other': other_labels,
    'dr_all': dr_all,
    'dr_lesions': dr_lesions,
    'dr_labels': dr_labels,
    'lesion_type': lesion_type,
    'lesion_distribution': lesion_distribution,
    'lesion_amount': lesion_amount,
    'lesion_size': lesion_size,
}

# lower than 0.95
insignificant_labels = [
    'age_related_macular_degeneration',
    'dr_detection_categories',
    'dr_grading_categories',
    'eyeside',
    'optic_disc_border',
    'optic_disc_color',
    'optic_disc_shape',
    'optic_disc_size',
    'peripapillary',
    'vessel_clarity',
    'vessel_color',
    'vessel_width'
]

augmentation_affected_labels = [
    'eyeside',
    'background',
    'optic_disc_color',
    'vessel_color',
    'vessel_clarity',
    'clarity'
]

def get_pkg_labels(label_names):
    return merge(*[label_collections[name] for name in label_names])

def remove_insignificant_labels(labels):
    new_labels = labels.copy()
    for insignificant_label in insignificant_labels:
        if insignificant_label in labels.keys():
            new_labels.pop(insignificant_label)
    return new_labels

def remove_augmentation_affected_labels(labels):
    new_labels = labels.copy()
    for affected_label in augmentation_affected_labels:
        if affected_label in labels.keys():
            new_labels.pop(affected_label)
    return new_labels
