def evaluate(model: torch.nn.Module, forward_pass_callback_args):
    """
     This is intended to be the user-defined model evaluation function. AIMET requires the above signature. So if the
     user's eval function does not match this signature, please create a simple wrapper.
     Use representative dataset that covers diversity in training data to compute optimal encodings.

    :param model: Model to evaluate
    :param forward_pass_callback_args: These argument(s) are passed to the forward_pass_callback as-is. Up to
            the user to determine the type of this parameter. E.g. could be simply an integer representing the number
            of data samples to use. Or could be a tuple of parameters or an object representing something more complex.
            If set to None, forward_pass_callback will be invoked with no parameters.
    """
    dummy_input = torch.randn(32, 3,320, 320).to(torch.device('cuda'))
    
    cfg = (voc320, voc512)[0]
    refinedet320 = VGGRefineDet(cfg['num_classes'], cfg)
    refinedet320.create_architecture()
    
    refinedet320.load_state_dict(torch.load('./output/vgg16_refinedet320_voc_120000.pth'))
    refinedet320.eval()
    with torch.no_grad():
        refinedet320(dummy_input)

def quantsim_example():

    AimetLogger.set_level_for_all_areas(logging.INFO)
    
    cfg = (voc320, voc512)[0]
    refinedet320 = VGGRefineDet(cfg['num_classes'], cfg)
    refinedet320.create_architecture()
    
    refinedet320.load_state_dict(torch.load('./output/vgg16_refinedet320_voc_120000.pth'))
    
    input_shape = (32, 3,320, 320)
    
    refinedet320.eval()
    
    refinedet320.cuda()
    input_shape = (32, 3,320, 320)
    dummy_input = torch.randn(input_shape).cuda()

    # Prepare model for Quantization SIM. This will automate some changes required in model definition for example
    # create independent modules for torch.nn.functional and reused modules
    prepared_model = prepare_model(refinedet320)

    # Instantiate Quantization SIM. This will insert simulation nodes in the model
    quant_sim = QuantizationSimModel(prepared_model, dummy_input=dummy_input,
                                     quant_scheme=QuantScheme.post_training_tf_enhanced,
                                     default_param_bw=8, default_output_bw=8
                                     #,config_file='../../TrainingExtensions/common/src/python/aimet_common/quantsim_config/'
                                                 #'default_config.json'
                                                 )

    # Compute encodings (min, max, delta, offset) for activations and parameters. Use representative dataset
    # roughly ~1000 examples
    quant_sim.compute_encodings(evaluate, forward_pass_callback_args=None)

    # QAT - Quantization Aware Training - Fine-tune the model fore few epochs to retain accuracy using train loop
    #data_loader = create_fake_data_loader(dataset_size=32, batch_size=16, image_size=input_shape[1:])
    #_ = train(quant_sim.model, data_loader)

    # Export the model which saves pytorch model without any simulation nodes and saves encodings file for both
    # activations and parameters in JSON format
    quant_sim.export(path='./', filename_prefix='quantized_refinedet320', dummy_input=dummy_input.cpu())