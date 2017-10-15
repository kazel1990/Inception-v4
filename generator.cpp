#include<stdio.h>
#include<string>
#include<vector>

constexpr int batch_size = 32;
int indent;
char buf[1<<8];

void print(const char * str)
{
    bool add_indent = false;
    for(int i=0;str[i];i++)
    {
        if(str[i] == '{') add_indent = true;
        if(str[i] == '}') indent--;
    }
    for(int i=0;i<indent;i++) printf("  ");
    if(add_indent) indent++;
    puts(str);
}

void in_out(std::vector<std::string> bot, std::string top)
{
    for(std::string str : bot)
    {
        sprintf(buf,"bottom: \"%s\"",str.c_str());
        print(buf);
    }
    sprintf(buf,"top: \"%s\"",top.c_str());
    print(buf);
}

void in_out(std::string bot, std::string top)
{
    sprintf(buf,"bottom: \"%s\"",bot.c_str());
    print(buf);
    sprintf(buf,"top: \"%s\"",top.c_str());
    print(buf);
}

//TODO: handle hyperparams outside
void convolution(std::string name, std::string bot, std::string top,
        int output, int pad, int kernel, int stride)
{
    print("layer {");
    sprintf(buf,"name: \"%s\"",name.c_str());
    print("type: \"Convolution\"");
    in_out(bot,top);
    print("param {");
    print("lr_mult: 1");
    print("decay_mult: 1");
    print("}");
    print("param {");
    print("lr_mult: 2");
    print("decay_mult: 0");
    print("}");

    print("convolution_param {");
    sprintf(buf,"num_output: %d",output);
    print(buf);
    sprintf(buf,"pad: %d",pad);
    print(buf);
    sprintf(buf,"kernel_size: %d",kernel);
    print(buf);
    sprintf(buf,"stride: %d",stride);
    print(buf);

    print("weight_filler {");
    print("type: \"xavier\"");
    print("std: 0.01");
    print("}");

    print("bias_filler {");
    print("type: \"constant\"");
    print("value: 0.2");
    print("}");

    print("}");

    print("}");
}

void batch_norm(std::string blob)
{
    print("layer {");
    sprintf(buf,"name: \"%s_bn\"",blob.c_str());
    print(buf);
    print("type: \"BatchNorm\"");
    in_out(blob,blob);
    print("batch_norm_param {");
    print("use_global_stats: false");
    print("}");

    print("}");
}

void scale(std::string blob)
{
    print("layer {");
    sprintf(buf,"name: \"%s_scale\"",blob.c_str());
    print("type: \"Scale\"");
    in_out(blob,blob);
    print("scale_param {");
    print("bias_term: true");
    print("}");
    print("}");
}

void relu(std::string blob)
{
    print("layer {");
    sprintf(buf,"name: \"%s_scale\"",blob.c_str());
    print("type: \"ReLU\"");
    in_out(blob,blob);
    print("}");
}

void pool(std::string name, std::string bot, std::string top,
        int kernel, int stride)
{
    sprintf(buf,"name: \"%s\"",name.c_str());
    print(buf);
    print("type: \"Pooling\"");
    in_out(bot, top);
    print("pooling_param {");
    print("pool: MAX");
    sprintf(buf,"kernel_size: %d",kernel);
    print(buf);
    sprintf(buf,"stride: %d",stride);
    print(buf);
    print("}");

    print("}");
}

void concat(std::string name, std::vector<std::string> bot, std::string top)
{
    print("layer {");
    sprintf(buf,"name: \"%s_scale\"",name.c_str());
    print(buf);
    print("type: \"Concat\"");
    in_out(bot, top);
}

void create_data()
{
    for(int i=0;i<=1;i++)
    {
        print("layer {");

        print("name: \"data\"");
        print("type: \"Data\"");
        print("top: \"data\"");
        print("top: \"label\"");

        print("include {");
        sprintf(buf,"phase: %s",i?"TEST":"TRAIN");
        print(buf);
        print("}");

        print("transform_param {");
        sprintf(buf,"mirror: %s",i?"false":"true");
        print(buf);
        print("crop_size: 299");
        print("mean_value: 104");
        print("mean_value: 117");
        print("mean_value: 123");
        print("}");

        print("data_param {");
        sprintf(buf,"source: \"examples/imagenet/ilsvrc12_%s_lmdb\"",
                i?"val":"train");
        print(buf);
        sprintf(buf,"batch_size: %d",batch_size);
        print(buf);
        print("backend: LMDB");
        print("}");

        print("}");
    }
}

void create_stem()
{
    std::string prv = "data", cur = "stem_conv1_3x3";
    auto norm = [](std::string str){
        batch_norm(str);
        scale(str);
        relu(str);
    };
    convolution(cur, prv, cur, 32, 0, 3, 2);
    norm(cur);

    prv = cur;
    cur = "stem_conv2_3x3";
    convolution(cur, prv, cur, 32, 0, 3, 1);
    norm(cur);

    prv = cur;
    cur = "stem_conv3_3x3";
    convolution(cur, prv, cur, 64, 1, 3, 1);
    norm(cur);

    prv = cur;
    std::string cur1 = "stem_inception1_pool";
    pool(cur1, prv, cur1, 3, 2);

    std::string cur2 = "stem_inception1_conv_3x3";
    convolution(cur2, prv, cur2, 96, 0, 3, 2);
    norm(cur2);

    cur = "stem_inseption1_concat";
    concat(cur, {cur1, cur2}, cur);
}

int main()
{
    freopen("train_val.prototxt","w",stdout);
    create_data();
    create_stem();
}
