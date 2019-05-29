import numpy as np
def complex_to_real(inp):
    Hr = np.real(inp)
    Hi = np.imag(inp)
    h1 = np.concatenate([Hr, -Hi], axis=2)
    h2 = np.concatenate([Hi,  Hr], axis=2)
    inp = np.concatenate([h1, h2], axis=1)
    return inp

def get_data(exp_name):
    if exp_name in ['accvscond']:
        H_dataset = np.load('/data/Mehrdad/channel_sequences_normalized.npy')[:,0]
        #H_dataset = np.load('/data/Mehrdad/channels_normalized.npy')[256:512]
        H_dataset = np.reshape(H_dataset, (-1, H_dataset.shape[-2], H_dataset.shape[-1]))
        H_dataset = complex_to_real(H_dataset)
        Hdataset_powerdB = 10. * np.log(np.mean(np.sum(H_dataset ** 2, axis=1))) / np.log(10.) 
        print(Hdataset_powerdB)

        train_data = H_dataset
        test_data = H_dataset

    if exp_name in ['test']:
        H_dataset = np.load('/data/Mehrdad/channel_sequences_normalized.npy')[0,0]
        #H_dataset = np.load('/data/Mehrdad/channels_normalized.npy')[256:512]
        H_dataset = np.reshape(H_dataset, (-1, H_dataset.shape[-2], H_dataset.shape[-1]))
        H_dataset = complex_to_real(H_dataset)
        Hdataset_powerdB = 10. * np.log(np.mean(np.sum(H_dataset ** 2, axis=1))) / np.log(10.) 
        print(Hdataset_powerdB)

        train_data = H_dataset
        test_data = H_dataset

    if exp_name in ['test2']:
        H_dataset = np.load('/data/Mehrdad/Mehrdads_channels_normalized.npy')[0,:]#[290,50]
        #H_dataset = np.load('/data/Mehrdad/channels_normalized.npy')[256:512]
        H_dataset = np.reshape(H_dataset, (-1, H_dataset.shape[-2], H_dataset.shape[-1]))
        H_dataset = complex_to_real(H_dataset)
        Hdataset_powerdB = 10. * np.log(np.mean(np.sum(H_dataset ** 2, axis=1))) / np.log(10.) 
        print(Hdataset_powerdB)

        train_data = H_dataset
        test_data = H_dataset

    if exp_name in ['test3']:
        #H_dataset = np.load('/data/Mehrdad/channel_sequences_normalized.npy')[0,0]
        H_dataset = np.load('/data/Mehrdad/channels_normalized.npy')#[256:512]
        H_dataset = np.reshape(H_dataset, (256,-1, 64, 16))[0,256]
        H_dataset = np.reshape(H_dataset, (-1, H_dataset.shape[-2], H_dataset.shape[-1]))
        H_dataset = complex_to_real(H_dataset)
        Hdataset_powerdB = 10. * np.log(np.mean(np.sum(H_dataset ** 2, axis=1))) / np.log(10.) 
        print(Hdataset_powerdB)

        train_data = H_dataset
        test_data = H_dataset

    if exp_name in ['exp2']:
        H_dataset = np.load('/data/Mehrdad/channel_sequences_normalized.npy')[:,0]
        H_dataset = np.reshape(H_dataset, (-1, H_dataset.shape[-2], H_dataset.shape[-1]))
        H_dataset = complex_to_real(H_dataset)
        Hdataset_powerdB = 10. * np.log(np.mean(np.sum(H_dataset ** 2, axis=1))) / np.log(10.) 
        print(Hdataset_powerdB)

        train_data = H_dataset[0:int(0.8*np.shape(H_dataset)[0])]
        test_data = H_dataset[int(0.8*np.shape(H_dataset)[0])+1:-1]

    if exp_name in ['exp5']:
        H_dataset = np.load('/data/Mehrdad/channel_sequences_normalized.npy')[0,:]
        H_dataset = np.reshape(H_dataset, (-1, H_dataset.shape[-2], H_dataset.shape[-1]))
        H_dataset = complex_to_real(H_dataset)
        Hdataset_powerdB = 10. * np.log(np.mean(np.sum(H_dataset ** 2, axis=1))) / np.log(10.) 
        print(Hdataset_powerdB)

        train_data = H_dataset[0:int(0.8*np.shape(H_dataset)[0])]
        test_data = H_dataset[int(0.8*np.shape(H_dataset)[0])+1:-1]

    if exp_name in ['exp6', 'exp9', 'exp10', 'exp11', 'exp12']:
        H_dataset = np.load('/data/Mehrdad/channel_sequences_normalized.npy')[0,0]
        H_dataset = np.reshape(H_dataset, (-1, H_dataset.shape[-2], H_dataset.shape[-1]))
        H_dataset = complex_to_real(H_dataset)
        Hdataset_powerdB = 10. * np.log(np.mean(np.sum(H_dataset ** 2, axis=1))) / np.log(10.) 
        print(Hdataset_powerdB)

        train_data = np.array([H_dataset[0]])
        test_data = np.array([H_dataset[0]])

    if exp_name in ['exp13', 'exp14']:
        H_dataset = np.load('/data/Mehrdad/channel_sequences_normalized.npy')[0,1]
        H_dataset = np.reshape(H_dataset, (-1, H_dataset.shape[-2], H_dataset.shape[-1]))
        H_dataset = complex_to_real(H_dataset)
        Hdataset_powerdB = 10. * np.log(np.mean(np.sum(H_dataset ** 2, axis=1))) / np.log(10.) 
        print(Hdataset_powerdB)

        train_data = np.array([H_dataset[0]])
        test_data = np.array([H_dataset[0]])

    if exp_name in ['exp15']:
        H_dataset = np.load('/data/Mehrdad/channel_sequences_normalized.npy')[789,65]
        H_dataset = np.reshape(H_dataset, (-1, H_dataset.shape[-2], H_dataset.shape[-1]))
        H_dataset = complex_to_real(H_dataset)
        Hdataset_powerdB = 10. * np.log(np.mean(np.sum(H_dataset ** 2, axis=1))) / np.log(10.) 
        print(Hdataset_powerdB)

    if exp_name in ['exp16']:
        H_dataset = np.load('/data/Mehrdad/channel_sequences_normalized.npy')[790,65]
        H_dataset = np.reshape(H_dataset, (-1, H_dataset.shape[-2], H_dataset.shape[-1]))
        H_dataset = complex_to_real(H_dataset)
        Hdataset_powerdB = 10. * np.log(np.mean(np.sum(H_dataset ** 2, axis=1))) / np.log(10.) 
        print(Hdataset_powerdB)
        train_data = np.array([H_dataset[0]])
        test_data = np.array([H_dataset[0]])

    if exp_name in ['exp17']:
        H_dataset = np.load('/data/Mehrdad/channel_sequences_normalized.npy')[800,65]
        H_dataset = np.reshape(H_dataset, (-1, H_dataset.shape[-2], H_dataset.shape[-1]))
        H_dataset = complex_to_real(H_dataset)
        Hdataset_powerdB = 10. * np.log(np.mean(np.sum(H_dataset ** 2, axis=1))) / np.log(10.) 
        print(Hdataset_powerdB)
        train_data = np.array([H_dataset[0]])
        test_data = np.array([H_dataset[0]])

    if exp_name in ['exp18']:
        H_dataset = np.load('/data/Mehrdad/channel_sequences_normalized.npy')[0:100,0]
        H_dataset = np.reshape(H_dataset, (-1, H_dataset.shape[-2], H_dataset.shape[-1]))
        H_dataset = complex_to_real(H_dataset)
        Hdataset_powerdB = 10. * np.log(np.mean(np.sum(H_dataset ** 2, axis=1))) / np.log(10.) 
        print(Hdataset_powerdB)
        train_data = H_dataset[0:int(0.8*np.shape(H_dataset)[0])]
        test_data = H_dataset[int(0.8*np.shape(H_dataset)[0])+1:-1]

    if exp_name in ['online2']:
        H_dataset = np.load('/data/Mehrdad/channel_sequences_normalized.npy')[0:100,0]
        H_dataset = np.reshape(H_dataset, (-1, H_dataset.shape[-2], H_dataset.shape[-1]))
        H_dataset = complex_to_real(H_dataset)
        Hdataset_powerdB = 10. * np.log(np.mean(np.sum(H_dataset ** 2, axis=1))) / np.log(10.) 
        print(Hdataset_powerdB)
        train_data = H_dataset[0:int(0.8*np.shape(H_dataset)[0])]
        test_data = H_dataset[int(0.8*np.shape(H_dataset)[0])+1:-1]

    if exp_name in ['online3']:
        H_dataset = np.load('/data/Mehrdad/channel_sequences_normalized.npy')[0:100,45]
        H_dataset = np.reshape(H_dataset, (-1, H_dataset.shape[-2], H_dataset.shape[-1]))
        H_dataset = complex_to_real(H_dataset)
        Hdataset_powerdB = 10. * np.log(np.mean(np.sum(H_dataset ** 2, axis=1))) / np.log(10.) 
        print(Hdataset_powerdB)
        train_data = H_dataset
        test_data = H_dataset

    if exp_name in ['last_sec']:
        H_dataset = np.load('/data/Mehrdad/channel_sequences_normalized.npy')
        #H_dataset = np.load('/data/Mehrdad/channels_normalized.npy')[256:512]
        H_dataset = np.reshape(H_dataset, (-1, H_dataset.shape[-2], H_dataset.shape[-1]))
        H_dataset = complex_to_real(H_dataset)
        Hdataset_powerdB = 10. * np.log(np.mean(np.sum(H_dataset ** 2, axis=1))) / np.log(10.) 
        print(Hdataset_powerdB)

        train_data = H_dataset
        test_data = H_dataset

    if exp_name in ['last_sec2']:
        H_dataset = np.load('/data/Mehrdad/channel_sequences_normalized.npy')[0:256,0]
        #H_dataset = np.load('/data/Mehrdad/channels_normalized.npy')[256:512]
        H_dataset = np.reshape(H_dataset, (-1, H_dataset.shape[-2], H_dataset.shape[-1]))
        H_dataset = complex_to_real(H_dataset)
        Hdataset_powerdB = 10. * np.log(np.mean(np.sum(H_dataset ** 2, axis=1))) / np.log(10.) 
        print(Hdataset_powerdB)

        train_data = H_dataset
        test_data = H_dataset

    if exp_name in ['last_sec3']:
        H_dataset = np.load('/data/Mehrdad/channel_sequences_normalized.npy')[0:256,0]
        #H_dataset = np.load('/data/Mehrdad/channels_normalized.npy')[256:512]
        H_dataset = np.reshape(H_dataset, (-1, H_dataset.shape[-2], H_dataset.shape[-1]))
        H_dataset = complex_to_real(H_dataset)
        Hdataset_powerdB = 10. * np.log(np.mean(np.sum(H_dataset ** 2, axis=1))) / np.log(10.) 
        print(Hdataset_powerdB)

        train_data = H_dataset
        test_data = H_dataset

    if exp_name in ['exp22', 'exp23', 'exp24','exp25', 'exp26', 'exp29']:
        H_dataset = np.load('/data/Mehrdad/channel_sequences_normalized.npy')[:,0]
        #H_dataset = np.load('/data/Mehrdad/channels_normalized.npy')[256:512]
        H_dataset = np.reshape(H_dataset, (-1, H_dataset.shape[-2], H_dataset.shape[-1]))
        H_dataset = complex_to_real(H_dataset)
        Hdataset_powerdB = 10. * np.log(np.mean(np.sum(H_dataset ** 2, axis=1))) / np.log(10.) 
        print(Hdataset_powerdB)

        train_data = H_dataset
        test_data = H_dataset

    if exp_name in ['exp27']:
        H_dataset = np.load('/data/Mehrdad/channel_sequences_normalized.npy')[4635,0]
        #H_dataset = np.load('/data/Mehrdad/channels_normalized.npy')[256:512]
        H_dataset = np.reshape(H_dataset, (-1, H_dataset.shape[-2], H_dataset.shape[-1]))
        H_dataset = complex_to_real(H_dataset)
        Hdataset_powerdB = 10. * np.log(np.mean(np.sum(H_dataset ** 2, axis=1))) / np.log(10.) 
        print(Hdataset_powerdB)

        train_data = H_dataset
        test_data = H_dataset

    if exp_name in ['exp28']:
        H_dataset = np.load('/data/Mehrdad/Mehrdads_channels_normalized.npy')[4439,0]
        H_dataset = np.reshape(H_dataset, (-1, H_dataset.shape[-2], H_dataset.shape[-1]))
        H_dataset = complex_to_real(H_dataset)
        Hdataset_powerdB = 10. * np.log(np.mean(np.sum(H_dataset ** 2, axis=1))) / np.log(10.) 
        print(Hdataset_powerdB)

        train_data = H_dataset
        test_data = H_dataset

    if exp_name in ['exp30']:
        H_dataset = np.load('/data/Mehrdad/Jakobs_channels_normalized_32.npy')[:,0]
        H_dataset = np.reshape(H_dataset, (-1, H_dataset.shape[-2], H_dataset.shape[-1]))
        H_dataset = complex_to_real(H_dataset)
        Hdataset_powerdB = 10. * np.log(np.mean(np.sum(H_dataset ** 2, axis=1))) / np.log(10.) 
        print(Hdataset_powerdB)

        train_data = H_dataset
        test_data = H_dataset

    if exp_name in ['test_new']:
        H_dataset = np.load('/data/vol/Mehrdads_channels_normalized2.npy')[570,90]
        #H_dataset = np.load('/data/Mehrdad/channels_normalized.npy')[256:512]
        H_dataset = np.reshape(H_dataset, (-1, H_dataset.shape[-2], H_dataset.shape[-1]))
        H_dataset = complex_to_real(H_dataset)
        Hdataset_powerdB = 10. * np.log(np.mean(np.sum(H_dataset ** 2, axis=1))) / np.log(10.) 
        print(Hdataset_powerdB)

        train_data = H_dataset
        test_data = H_dataset

    if exp_name in ['errorProp_Nt32']:
        H_dataset = np.load('/data3/Jakobs_channels_normalized_32.npy')
        #H_dataset = np.load('/data/Mehrdad/channels_normalized.npy')[256:512]
        H_dataset = np.reshape(H_dataset, (-1, H_dataset.shape[-2], H_dataset.shape[-1]))
        H_dataset = complex_to_real(H_dataset)
        Hdataset_powerdB = 10. * np.log(np.mean(np.sum(H_dataset ** 2, axis=1))) / np.log(10.) 
        print(Hdataset_powerdB)

        train_data = H_dataset
        test_data = H_dataset

    if exp_name in ['errorProp']:
        H_dataset = np.load('/data2/Mehrdads_channels_normalized2.npy')
        #H_dataset = np.load('/data/Mehrdad/channels_normalized.npy')[256:512]
        H_dataset = np.reshape(H_dataset, (-1, H_dataset.shape[-2], H_dataset.shape[-1]))
        H_dataset = complex_to_real(H_dataset)
        Hdataset_powerdB = 10. * np.log(np.mean(np.sum(H_dataset ** 2, axis=1))) / np.log(10.) 
        print(Hdataset_powerdB)

        train_data = H_dataset
        test_data = H_dataset

    if exp_name in ['errorPropIid']:
        H_dataset = np.load('/data3/H_iid.npy')
        #H_dataset = np.load('/data/Mehrdad/channels_normalized.npy')[256:512]
        H_dataset = np.reshape(H_dataset, (-1, H_dataset.shape[-2], H_dataset.shape[-1]))
        H_dataset = complex_to_real(H_dataset)
        Hdataset_powerdB = 10. * np.log(np.mean(np.sum(H_dataset ** 2, axis=1))) / np.log(10.) 
        print(Hdataset_powerdB)

        train_data = H_dataset
        test_data = H_dataset

    if exp_name in ['errorPropH2', 'errorPropH4', 'errorPropH8', 'errorPropH16', 'errorPropH32', 'errorPropH64']:
        H_dataset = np.load('/data2/H%s.npy'%exp_name.lstrip('errorPropH'))
        H_dataset = np.reshape(H_dataset, (-1, H_dataset.shape[-2], H_dataset.shape[-1]))
        H_dataset = complex_to_real(H_dataset)
        Hdataset_powerdB = 10. * np.log(np.mean(np.sum(H_dataset ** 2, axis=1))) / np.log(10.) 
        print(Hdataset_powerdB)

        train_data = H_dataset
        test_data = H_dataset

    return train_data, test_data, Hdataset_powerdB
