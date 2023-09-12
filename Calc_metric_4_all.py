#========================
#
#========================

# This script calculate the strucural similarity defined by a certain metric between images
#
# use skimage to do the work

#========================
#        import
#========================

import common as c

#========================
#
#========================

measurements = ['structural_similarity_index', 'mean_squared_error', 'peak_signal_noise_ratio']
image_cutoff = [20 , 25]

#========================
#
#========================


# ---------------
# read in all images

images = c.prepare_3D_image_4_patient(mask=True)
images.find_patient_images()
images.check_patient_images()


# loop over all patients
names = {
    'pos_neg': ['positive_names', 'negative_names'],
    'positive_names': images.daughter_dirs['positive_names'],
    'negative_names':images.daughter_dirs['negative_names']
}

results = {'names':[]}

i = 1
for pos_neg in names['pos_neg']:
    for name in names[pos_neg]:

        print('\nComputing metrics for %2i-th patient %s ...' %(i, name))

        results['names'].append(name)

        patient = {
            'pos_neg': pos_neg,
            'patient_name': name,
            'image_cutoff': image_cutoff
        }

        image_patient = images.read_images_4_patient(patient)

        results_patient = {
            'patient_name':name,
            'recon_info':[]
        }

        # compute metrics
        # loop over 1st and 2nd recon methods
        for recon_1 in image_patient['recon_method']:
            for recon_2 in image_patient['recon_method']:

                print('\n  for %s vs %s' %(recon_1[1], recon_2[1]))

                recon_info = recon_1[1]+' v '+recon_2[1]
                results_patient['recon_info'].append(recon_info)
                results_patient[recon_info] = {'recon_info':recon_info}

                # loop over all measures
                str_tmp = '  '
                for measure_name in measurements:

                    measure = c.select_measure(measure_name)

                    if measure_name != 'mean_squared_error':
                        measure_tmp = measure( image_patient[recon_1[1]], image_patient[recon_2[1]], data_range=( image_patient[recon_1[1]].max() - image_patient[recon_1[1]].min() ) )

                    else:
                        measure_tmp = measure( image_patient[recon_1[1]], image_patient[recon_2[1]] )

                    results_patient[recon_info][measure_name] = measure_tmp
                    str_tmp += '  %s = %9.6f ,' %(measure_name, measure_tmp)

                print(str_tmp)

        results[name] = results_patient
        file_names = c.write_patient_results(results_patient, image_patient['recon_method'])

        with open('results_file.txt', 'a') as f:
            for fn in file_names:
                f.write('\n'+fn)
            f.close()

        print('  done')

        i += 1
        # if i > 3:
        #     quit()


print(results)



