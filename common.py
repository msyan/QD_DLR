#========================
#
#========================

# This script contains useful functions

#========================
#        import
#========================

import os
import numpy as np
import skimage
import skimage.metrics as skm
import matplotlib.pyplot as plt
import pydicom

#========================
#        define
#========================

class prepare_3D_image_4_patient():

    def __init__(self, mask=True):

        self.mother_path = r"Path/to/images"

        # this negative patient doesn't have 2min DLR images
        self.tricky_patient = ['阴性组DLR 2min', '20220623', 'WANG XUE QIN_6307416_113257']

        # whether to mask images
        self.mask = mask

    def find_patient_images(self):

        print('\nFinding directories of images for all patients...')

        # find all groups
        tmp_daughter_dirs = os.listdir(self.mother_path)
        self.daughter_dirs = {
            "daughter_folder":tmp_daughter_dirs
        }

        self.daughter_dirs['positive_names'] = []
        self.daughter_dirs['negative_names'] = []
        self.daughter_dirs['20220208[925]'] = []

        # loop over all groups
        for dd in tmp_daughter_dirs:

            md_path = self.mother_path + "\\" + dd
            tmp_dd_dirs = os.listdir(md_path)

            # loop over all dates
            for ddd in tmp_dd_dirs:

                mdd_path = md_path + "\\" + ddd
                tmp_ddd_dirs = os.listdir(mdd_path)

                # loop over all patients
                for dddd in tmp_ddd_dirs:

                    mddd_path = mdd_path + "\\" + dddd
                    tmp_dddd_dirs = os.listdir(mddd_path)

                    # initiate patient names in each group
                    if dd == "阳性组DLR 1min+OSEM":

                        self.daughter_dirs['positive_names'].append(ddd+"\\"+dddd)
                        self.daughter_dirs[ddd+"\\"+dddd] = {'recon_method':[]}

                    elif dd == "阴性组DLR 1min+OSEM（缺6307416）":

                        self.daughter_dirs['negative_names'].append(ddd+"\\"+dddd)
                        self.daughter_dirs[ddd+"\\"+dddd] = {'recon_method':[]}

                    elif dd == "20220208[925]":

                        self.daughter_dirs['20220208[925]'].append(ddd+"\\"+dddd)
                        self.daughter_dirs[ddd+"\\"+dddd] = {'recon_method':[]}

                    # loop over all recon. methods
                    for ddddd in tmp_dddd_dirs:

                        mdddd_path = mddd_path + "\\" + ddddd
                        tmp_ddddd_dirs = os.listdir(mdddd_path)

                        # skip the tricky patient
                        if (dd==self.tricky_patient[0]) and (ddd==self.tricky_patient[1]) and (dddd==self.tricky_patient[2]):
                            print('  we have passed: %s/%s,' %(ddd, dddd) )
                            print('    due to the lack of its %s image' %(dd))
                        else:
                            self.daughter_dirs[ddd+"\\"+dddd][ddddd] = tmp_ddddd_dirs
                            self.daughter_dirs[ddd+"\\"+dddd]['recon_method'].append([dd, ddddd])                    

        print('  done')

    def check_patient_images(self):

        print('\nChecking directories of images for all patients...')

        i = 1
        for neg_pos in ['positive_names', 'negative_names']:

            for patient_name in self.daughter_dirs[neg_pos]:

                if neg_pos == 'positive_names':
                    str_tmp = ' %2i - positive patient: %40s has  %1i  recon. method available - ' %(i, patient_name, len(self.daughter_dirs[patient_name]['recon_method']))
                else:
                    str_tmp = ' %2i - negative patient: %40s has  %1i  recon. method available - ' %(i, patient_name, len(self.daughter_dirs[patient_name]['recon_method']))

                for recon_method in self.daughter_dirs[patient_name]['recon_method']:

                    str_tmp += 'recon. method %26s  has  %3i  images , ' %(recon_method[1], len(self.daughter_dirs[patient_name][recon_method[1]]))

                num_images = [len(self.daughter_dirs[patient_name][recon_method[1]]) for recon_method in self.daughter_dirs[patient_name]['recon_method']]
                num_images_recon = zip(num_images, self.daughter_dirs[patient_name]['recon_method'])
                min_images_recon = min(num_images_recon)
                self.daughter_dirs[patient_name]['key_recon_method'] = min_images_recon[1]
        
                str_tmp += ' - should use %12s' %(self.daughter_dirs[patient_name]['key_recon_method'][1])
        
                print(str_tmp)

                i += 1   

        print('  done')

    # sort directories of a certain patient
    # by cutting off several first and last images
    def sort_directoy_4_patient(self, patient:dict):

        """ 
        patient = {
            'pos_neg': 'positive_names' or 'negative_names',
            'patient_name': string,
            image_cutoff: list:[cut_on_first_images, cut_on_last_images]
        }
        """

        key_recon_method = self.daughter_dirs[patient['patient_name']]['key_recon_method'][1]

        directories_used = {
            'mother_path': self.mother_path,
            'pos_neg': patient['pos_neg'],
            'patient_name': patient['patient_name'],
            'recon_method': self.daughter_dirs[patient['patient_name']]['recon_method'],
            'images':self.daughter_dirs[patient['patient_name']][key_recon_method][patient['image_cutoff'][0] : -patient['image_cutoff'][1]]
        }

        return directories_used

    def construct_path_2_image(self, path_info):
        path = self.mother_path + "\\" + path_info['recon_method'][0] + "\\" + path_info['patient_name'] + "\\" + path_info['recon_method'][1] + "\\" + path_info['dicom_file']
        return path

    # read in all requested images for a patient
    def read_images_4_patient(self, patient):

        print('\nReading in all dicom images for patient %s ...' %(patient['patient_name']))

        directories_used = self.sort_directoy_4_patient(patient)

        images_3D_recons = {
            'patient_name':patient['patient_name'],
            'pos_neg': patient['pos_neg'],
            'recon_method':directories_used['recon_method']
        }

        # loop over recon. methods
        for recon_method in directories_used['recon_method']:

            print('  reading image reconstructed by %s method ...' %(recon_method[1]))

            images_3D_recons_tmp = []

            # loop over all images after first/last cut-off
            for dicom_image in directories_used['images']:
            
                path_info = {
                    'recon_method': recon_method,
                    'patient_name': directories_used['patient_name'],
                    'dicom_file': dicom_image
                }

                path = self.construct_path_2_image(path_info)
                image_tmp = float_image( parse_dicom(path) )

                if self.mask:
                    masked_image = mask_image(image_tmp)
                    images_3D_recons_tmp.append( masked_image )
                else:
                    images_3D_recons_tmp.append( image_tmp )

                images_3D_recons[recon_method[1]] = np.stack(images_3D_recons_tmp)

        print('  done')

        return images_3D_recons

# ---------------

# erase value of pixels where the value is below the lower bound set by hand
# make sure the image is a numpy array
def mask_image(image, lower_bound=1E-3):

    masked_image = np.ma.masked_values(image, 0.0, atol=1E-3)
    filled_image = masked_image.filled(fill_value=0.0)

    return filled_image


def parse_dicom(image_path, only_pixel=True):

    ds = pydicom.dcmread(image_path)

    if only_pixel:
        return ds.pixel_array
    else:
        return ds

def float_image(py_image):

    floated_image = skimage.img_as_float(py_image)

    return floated_image

def select_measure(measure_name):

    if measure_name == 'structural_similarity_index':
        # print('use %s to measure the similarity' %(measure_name))
        measurement = skm.structural_similarity

    elif measure_name == 'peak_signal_noise_ratio':
        # print('use %s to measure the similarity' %(measure_name))
        measurement = skm.peak_signal_noise_ratio

    elif measure_name == 'mean_squared_error':
        # print('use %s to measure the similarity' %(measure_name))
        measurement = skm.mean_squared_error

    return measurement

def reconstruct_name(name):

    name_list = name.split('\\')
    name_list_2 = name_list[1].split()

    new_name = ''
    for n in name_list_2:
        new_name += n + '_'
    new_name += name_list[0]

    return new_name

def write_patient_results(results_patient, recon_methods, measurements=['structural_similarity_index', 'mean_squared_error', 'peak_signal_noise_ratio']):

    file_names = []

    # loop over all measures
    for measure_name in measurements:

        file_name = reconstruct_name(results_patient['patient_name'])+'_'+measure_name+'.txt'
        file_names.append(file_name)

        with open(file_name, 'w') as f:

            str_tmp = ''
            for irm, rm in enumerate(recon_methods):
                str_tmp += '  recon. method-%i: %s ,' %(irm+1, rm[1])
            f.write(str_tmp)

            str_tmp = '\n         '
            for i in range(len(recon_methods)):
                str_tmp += '  %8s%i' %('recon.-', i+1)
            f.write(str_tmp)

            # loop over 1st and 2nd recon methods
            for i in range(len(recon_methods)):

                str_tmp = '\n recon.-%i' %(i+1)
                for j in range(len(recon_methods)):

                    recon_info = recon_methods[i][1] + ' v ' + recon_methods[j][1]
                    str_tmp += '  %9.6f' %(results_patient[recon_info][measure_name])
                f.write(str_tmp)

            f.close()

    return file_names

# ---------------

def plot_4_images(images):

    fig, axes = plt.subplots(2,2, figsize=(18, 8.5), sharex=True, sharey=True)
    ax = axes.ravel()

    for iimg, image in enumerate(images):

        ax[iimg].set_title(image['title'])
        ax[iimg].imshow(image['image'], cmap=plt.cm.Greys_r)

    plt.show()


def compare_recon_images(image_1, image_2):

    """
    image_1 or image_2:
        {
            'image': grey image made of lists,
            'title': the titile of this image
        }
    """

    fig, axes = plt.subplots(1,2, figsize=(18, 8.5), sharex=True, sharey=True)
    ax = axes.ravel()

    ax[0].set_title(image_1['title'])
    ax[0].imshow(image_1['image'], cmap=plt.cm.Greys_r)

    ax[1].set_title(image_2['title'])
    ax[1].imshow(image_2['image'], cmap=plt.cm.Greys_r)

    plt.show()

def hist_index(measures, measure_name, title):

    plt.hist(measures, bins=15)
    plt.xlabel('image counts')
    plt.ylabel(measure_name)
    plt.grid(True)
    plt.savefig(title+'.pdf')
    plt.show()

def plt_array_index(measure, measure_name, title):

    array = [i for i in range(1, len(measure)+1)]

    plt.plot(array, measure)
    plt.xlabel('image index')
    plt.ylabel(measure_name)
    plt.grid(True)
    plt.savefig(title+'.pdf')
    plt.show()


def plot_info_summary(results):
    
    # loop over all measurements
    for measure_name in results['indices']:

        measure = results[measure_name]

        hist_index(measure, measure_name, 'hist_'+results['info']+'_'+measure_name)
        plt_array_index(measure, measure_name, 'array_'+results['info']+'_'+measure_name)



