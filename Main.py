# this is the main method for my entire program.
import itertools
import math
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.utils.data

from Utility import feature_selection as fs
from Utility import ngram
from Utility.opcodeseq_creator import run_opcode_seq_creation
from models.BNN import train, test
import models.DBN as dbn
import xml.etree.ElementTree as et
import lxml.etree as ET

##
##  (opt) INPUT: apk dir, tmp dir, opcode dir, fcbl dir, model dir
##
##
sys.path.insert(1, os.path.join(sys.path[0], '../..'))


def converttoopcode(apkdir, tmpdir, opdir, suplib):
    """"
    :param apkdies. default is no
    output: None
    ir: the directory to get apks from
    :param tmpdir: the temporary dir to store while converting to opcode
    :param opdir: the directory to story the opcodeseq
    :param suplib: whether to use support librar
    This function takes apkdir, tmpdir, opdir and suplib and convert all apk's in apkdir to opcode sequences
    in opdir. See opcode based malware detection for more details.
    """
    # Simple function. Just run opcode seq creation
    run_opcode_seq_creation.main(apkdir=apkdir, tmpdir=tmpdir, opdir=opdir, suplib="")


def getdataset(dire, nograms):
    """
    :param dire: dir of texts to be converted to ngrams (opcodeseq file)
    :param nograms: the number (n) of ngrams to be split into
    output: None
    """
    # appends all txtfiles in array
    txt = []
    num = 0
    # Second param doesn't matter right now, it's just for easily customizability
    # Get all opseq locations
    for txtfile in os.listdir(dire):
        if os.path.isfile(os.path.join(dire, txtfile)):
            txt.append(txtfile)
    nofiles = len(txt)
    perms = pd.DataFrame(columns=['Malware'])
    print("Filling dataframe with counts")
    # Go into opseq locations
    for txt_file in txt:
        # Get all possible ngrams for a file
        if num % 40 == 0 and num != 0:
            print(str(num) + " of " + str(nofiles) + "files have been finished.")
        tmp, nowords = getngrams(txt_file, dire, nograms)
        # if none of first two are digits, it's benign
        if not (txt_file[0].isdigit()) and not (txt_file[1].isdigit()):
            perms.at[num, 'Malware'] = 0
        else:
            perms.at[num, 'Malware'] = 1
        for gram in tmp:
            perms.at[num, gram] += (1 / (nowords))
        num += 1
    print("Extracting features")
    # Reduces it to 2048 features
    features = feature_extraction(perms)
    print("Creating input for model")
    nninput = pd.DataFrame(index=[np.arange(nofiles)], columns=[features.columns])
    for col in features:
        nninput[[col]] = perms[[col]].values
    print(nninput)
    dataset = torch.tensor(nninput[:].values).double()
    dataset = dataset[torch.randperm(dataset.size()[0])]
    print("Input for model created")
    return dataset

def getperms():

def feature_extraction(arr):
    # Drops malware column
    X = arr.drop('Malware', axis=1)
    # Initialize array to store values
    val = pd.DataFrame(index=[0], columns=X.columns, dtype="float64")
    # Gets malware column
    Y = arr['Malware']
    # Gets SU of all features
    for feature in X:
        Fi = X[feature]
        val.loc[0, feature] = fs.SU(Fi, Y)
    # Sort it based on index 0 value 0 (.at[0,:]
    result = val.sort_values(0, axis=1, ascending=False)
    # Get top 4096 values
    result = result.iloc[:, 0:2048]
    result[['Malware']] = arr[['Malware']]
    return result


def getngrams(fname, dire, nograms):
    ngram_generator = ngram.Ngram()
    ngramcomplete = []
    """
    :param fname: file name to be read
    :param nograms: number of grams wanted to be partitioned into
    :return: list of all grams in the file
    """
    # open file
    fname = os.path.join(dire, fname)
    with open(fname, mode="r") as bigfile:
        # read it as txt
        reader = bigfile.read()
        # removes newlines as to not mess with n-gram
        reader = reader.replace('\\', "")
        reader = reader.replace('\n', "")
        # append list to list
        nowords = len(reader) // 2
        ngramcomplete = ngram_generator.letterNgram(reader, nograms)
    return ngramcomplete, nowords


###############################################
# UNUSED, decided to use another one as data  #
# is just too sparse. I would look into using #
# elmo. Issue is takes forever to train.      #
###############################################
def converttofcbl(csvdir, nofeatures):
    """"
    :param csvdir: the directory of the csv files
    :param nofeatures: the number of features you want the csv to be reduced to
    output: numpy array with proper attributes

    This function takes a csvdir and the nofeatures and takes all csv's in csvdir and reduces it to the number of
    features using entropy. See Fast Correlation Based Entropy for data dimensionality reduction for more i2nformation.
    """
    data = pd.read_csv(csvdir)
    # fs.fcbf(data, label, threshold = 0.2, base=2)


def parsepermissiondirectory(dire):
    #Get list of all permissions (166 of them)
    perms = getpermissions()
    txt = []
    i = 0
    #get file names
    for xmlfile in os.listdir(dire):
        if os.path.isfile(os.path.join(dire, xmlfile)):
            txt.append(xmlfile)
    #get number of files
    nofiles = len(txt)
    #create dataframe
    perm = pd.DataFrame(0, columns=perms + ['Malware'], index=np.arange(nofiles))

    for xmlfile in txt:
        # get malware flag
        if (i % 50 == 0):
            print("On file " + str(i))
        if not (xmlfile[0].isdigit()) and not (xmlfile[1].isdigit()) and not (xmlfile[2].isdigit()) and not (xmlfile[3].isdigit()) \
                and not (xmlfile[4].isdigit()) and not (xmlfile[5].isdigit()):
            malwareflag=0
        else:
            malwareflag=1
        # get permission array
        permfile = parsepermissionfile(os.path.join(dire, xmlfile))
        # append to big array
        perm.at[i, 'Malware'] = malwareflag
        for permission in permfile:
            if permission in perm.columns:
                perm.at[i, permission] = 1
        i = i + 1
    return perm


def parsepermissionfile(f):
    """
    :param f: file to parse
    :return: list of all perms in this file
    """
    # get list of all possible permissions
    t = getpermissions()
    # list to add to
    listofperms = []
    # edge case where no manifest
    if os.stat(f).st_size == 0:
        return []
    # parse tree
    parser = ET.XMLParser(recover=True)
    tree = ET.parse(f, parser=parser)

    et.register_namespace('android', 'http://schemas.android.com/apk/res/android')
    root = tree.getroot()
    #loop through all permissions
    for node in root.findall('uses-permission'):
        # get all uses-permissions
        list = node.items()
        perm = list[0][1]
        # make sure it's android.permission.PERMISSION
        if "android.permission." in perm:
            # get only permission, append to listofperms
            onlyperm = perm.replace("android.permission.", "")
            listofperms.append(onlyperm)
    # replace values in dataframe
    return listofperms


def getpermissions():
    t = ['ACCEPT_HANDOVER', 'ACCESS_BACKGROUND_LOCATION', 'ACCESS_CHECKIN_PROPERTIES', 'ACCESS_CHECKIN_PROPERTIES',
         'ACCESS_COARSE_LOCATION', 'ACCESS_FINE_LOCATION', 'ACCESS_LOCATION_EXTRA_COMMANDS',
         'ACCESS_MEDIA_LOCATION', 'ACCESS_NETWORK_STATE', 'ACCESS_NOTOFICATION_POLICY', 'ACCESS_WIFI_STATE',
         'ACCOUNT_MANAGER', 'ACTIVITY_RECOGNITION', 'ADD_VOICEMAIL', 'ANSWER_PHONE_CALLS', 'BATTERY_STATE',
         'BIND_ACCESSIBILITY_SERVICE'
        , 'BIND_APPWIDGET', 'BIND_AUTOFILL_SERVICE', 'BIND_CALL_REDIRECTION_SERVICE',
         'BIND_CARRIER_MESSAGING_CLIENT_SERVICE', 'BIND_CARRIER_MESSAING_SERVICE', 'BIND_CARRIER_SERVICES',
         'BIND_CHOOSER_TARGET_SERVICE', 'BIND_CONDITION_PROVIDER_SERVICE', 'BIND_CONTROLS'
        , 'BIND_DEVICE_ADMIN', 'BIND_DREAM_SERVICE', 'BIND_INCALL_SERVICE', 'BIND_INPUT_METHOD',
         'BIND_MIDI_DEVICE_SERVICE', 'BIND_NFC_SERVICE', 'BIND_NOTIFICATION_LISTENER_SERVICE', 'BIND_PRINT_SERVICE',
         'BIND_QUICK_ACCESS_WALLET_SERVICE',
         'BIND_QUICK_SETTINGS_TILE', 'BIND_REMOTEVIEWS', 'BIND_SCREENING_SERVICE', 'BIND_TELECOM_CONNECTION_SERVICE',
         'BIND_TEXT_SERVICE', 'BIND_TV_INPUT', 'BIND_VISUAL_VOICEMAIL_SERVICE', 'BIND_VOICE_INTERACTION',
         'BIND_VPN_SERVICE',
         'BIND_VR_LISTENER_SERVICE', 'BIND_WALLPAPER', 'BLUETOOTH', 'BLUETOOTH_ADMIN', 'BLUETOOTH_PRIVILEGED',
         'BODY_SENSORS', 'BROADCAST_PACKAGE_REMOVED', 'BROADCAST_SMS', 'BROADCAST_STICKY', 'BROADCAST_WAP_PUSH',
         'CALL_COMPANION_APP',
         'CALL_PHONE', 'CALL_PRIVILEGED', 'CAMERA', 'CAPTURE_AUDIO_OUTPUT', 'CHANGE_COMPONENT_ENABLED_STATE',
         'CHANGE_CONFIGURATION', 'CHANGE_NETWORK_STATE', 'CHANGE_WIFI_MULTICAST_STATE', 'CHANGE_WIFI_STATE',
         'CLEAR_APP_CACHE', 'CONTROL_LOCATION_UPDATES', 'DELETE_CACHE_FILES', 'DELETE_PACKAGES', 'DIAGNOSTIC',
         'DISABLE_KEYGUARD', 'DUMP', 'EXPAND_STATUS_BAR', 'FACTORY_TEST', 'FOREGROUND_SERVICE', 'GET_ACCOUNTS',
         'GET_ACCOUNTS_PRIVILEGED', 'GET_PACKAGE_SIZE', 'GET_TASKS', 'GLOBAL_SEARCH', 'INSTALL_LOCATION_PROVIDER',
         'INSTALL_PACKAGES', 'INSTALL_SHORTCUT', 'INSTANT_APP_FOREGROUND_SERVICE', 'INTERACT_ACROSS_PROFILES',
         'INTERNET',
         'KILL_BACKGROUND_PROCESSES', 'LOADER_USAGE_STATS', 'LOCATION_HARDWARE', 'MANAGE_DOCUMENTS',
         'MANAGE_EXTERNAL_STORAGE', 'MANAGE_OWN_CALLS', 'MASTER_CLEAR', 'MEDIA_CONTENT_CONTROL',
         'MODIFY_AUDIO_SETTINGS',
         'MODIFY_PHONE_STATE', 'MOUNT_FORMAT_FILESYSTEMS', 'MOUNT_UNMOUN_FILESYSTEMS', 'NFC',
         'NFC_PREFERRED_PAYMENT_INFO', 'NFC_TRANSACTION_EVENT', 'PACKAGE_USAGE_STATS', 'PERSISTENT_ACTIVITY',
         'PROCESS_OUTGOING_CALLS', 'QUERY_ALL_PACKAGES',
         'READ_CALENDER', 'READ_CALL_LOG', 'READ_CONTACTS', 'READ_EXTERNAL_STORAGE', 'READ_INPUT_STATE', 'READ_LOGS',
         'READ_PHONE_NUMBERS', 'READ_PHONE_STATE', 'READ_PRECISE_PHONE_STATE', 'READ_SMS', 'READ_SYNC_SETTINGS',
         'READ_VOICEMAIL', 'REBOOT', 'RECEIVE_BOOT_COMPLETED', 'RECEIVE_MMS', 'RECEIVE_SMS', 'RECEIVE_WAP_PUSH',
         'RECORD_AUDIO', 'REORDER_TASKS', 'REQUEST_COMPANION_RUN_IN_BACKGROUND',
         'REQUEST_COMPANION_USE_DATA_IN_BACKGROUND',
         'REQUEST_DELETE_PACKAGES', 'REQUEST_IGNORE_BATTERY_OPTIMIZATIONS', 'REQUEST_INSTALL_PACKAGES',
         'REQUEST_PASSWORD_COMPLEXITY', 'RESTART_PACKAGES', 'SEND_RESPOND_VIA_MESSAGE', 'SEND_SMS', 'SET_ALARM',
         'SET_ALWAYS_FINISH', 'SET_ANIMATION_SCALE',
         'SET_DEBUG_APP', 'SET_PREFERRED_APPLICATIONS', 'SET_PROCESS_LIMIT', 'SET_TIME', 'SET_TIME_ZONE',
         'SET_WALLPAPER', 'SET_WALLPAPER_HINTS', 'SIGNAL_PERSISTENT_PROCESSES', 'SMS_FINANCIAL_TRANSACTIONS',
         'START_VIEW_PERMISSION_USAGE',
         'STATUS_BAR', 'SYSTEM_ALERT_WINDOW', 'TRANSMIT_IR', 'UNINSTALL_SHORTCUT', 'UPDATE_DEVICE_STATS',
         'USE_BIOMETRIC', 'USE_FINGERPRINT', 'USE_FULL_SCREEN_INTENT', 'USE_SIP', 'VIBRATE', 'WAKE_LOCK',
         'WRITE_APN_SETTINGS', 'WRITE_CALENDER',
         'WRITE_CALL_LOG', 'WRITE_CONTACTS', 'WRITE_EXTERNAL_STORAGE', 'WRITE_GSERVICES', 'WRITE_SECURE_SETTINGS',
         'WRITE_SETTINGS', 'WRITE_SYNC_SETTINGS', 'WRITE_VOICEMAIL']
    return t


def dbndataset():
    f = os.path.join("D:\\5th\Honours\Code\manidataset\\benign", "com.coafit.apk.smali.xml")
    p = parsepermissiondirectory(f)


if __name__ == "__main__":
    dataset = getdataset("D:\\5th\Honours\Code\opdataset\mixed", 4)
    train(dataset)
    #model = dbn.DBN()
    #dataset = parsepermissiondirectory("D:\\5th\Honours\Code\manidataset\\mixed")
    #dataset = torch.tensor(dataset[:].values).double()
    #dataset = dataset[torch.randperm(dataset.size()[0])]
    #model.train_static(dataset)
    #train(dataset)