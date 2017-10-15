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
        int output, int pad_h, int pad_w, int kernel_h, int kernel_w,
        int stride)
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
    if(pad_h == pad_w)
    {
        sprintf(buf,"pad: %d",pad_h);
        print(buf);
    }
    else
    {
        sprintf(buf,"pad_h: %d",pad_h);
        print(buf);
        sprintf(buf,"pad_w: %d",pad_w);
        print(buf);
    }
    if(kernel_h == kernel_w)
    {
        sprintf(buf,"kernel_size: %d",kernel_h);
        print(buf);
    }
    else
    {
        sprintf(buf,"kernel_h: %d",kernel_h);
        print(buf);
        sprintf(buf,"kernel_w: %d",kernel_w);
        print(buf);
    }
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

void convolution(std::string name, std::string bot, std::string top,
        int output, int pad, int kernel, int stride)
{
    convolution(name, bot, top, output, pad, pad, kernel, kernel, stride);
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
        std::string type, int kernel, int stride, int pad=0)
{
    print("layer {");
    sprintf(buf,"name: \"%s\"",name.c_str());
    print(buf);
    print("type: \"Pooling\"");
    in_out(bot, top);
    print("pooling_param {");
    sprintf(buf,"pool: %s",type.c_str());
    print(buf);
    sprintf(buf,"kernel_size: %d",kernel);
    print(buf);
    sprintf(buf,"stride: %d",stride);
    print(buf);
    if(pad)
    {
        sprintf(buf,"pad: %d",pad);
        print(buf);
    }
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
    print("}");
}

void norm(std::string str)
{
    batch_norm(str);
    scale(str);
    relu(str);
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

std::string create_stem(std::string prv)
{
    std::string cur = "stem_conv1_3x3";
    std::string cur1, cur2, prv1, prv2;
    
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
    cur1 = "stem_inception1_pool";
    pool(cur1, prv, cur1, "MAX", 3, 2);

    cur2 = "stem_inception1_conv_3x3";
    convolution(cur2, prv, cur2, 96, 0, 3, 2);
    norm(cur2);

    cur = "stem_inception1_concat";
    concat(cur, {cur1, cur2}, cur);

    prv = cur;
    cur1 = "stem_inception2_conv1_1_1x1";
    convolution(cur1, prv, cur1, 64, 0, 1, 1);
    norm(cur1);

    prv1 = cur1;
    cur1 = "stem_inception2_conv1_2_3x3";
    convolution(cur1, prv1, cur1, 96, 0, 3, 1);
    norm(cur1);

    cur2 = "stem_inception2_conv2_1_1x1";
    convolution(cur2, prv, cur2, 64, 0, 1, 1);
    norm(cur2);

    prv2 = cur2;
    cur2 = "stem_inception2_conv2_2_7x1";
    convolution(cur2, prv2, cur2, 64, 3, 0, 7, 1, 1);
    norm(cur2);

    prv2 = cur2;
    cur2 = "stem_inception2_conv2_3_1x7";
    convolution(cur2, prv2, cur2, 64, 0, 3, 1, 7, 1);
    norm(cur2);

    prv2 = cur2;
    cur2 = "stem_inception2_conv2_4_3x3";
    convolution(cur2, prv2, cur2, 96, 0, 3, 1);
    norm(cur2);

    cur = "stem_inception2_concat";
    concat(cur, {cur1, cur2}, cur);

    prv = cur;

    cur1 = "stem_inception3_conv_3x3";
    convolution(cur1, prv, cur1, 192, 0, 3, 1);
    norm(cur1);

    cur2 = "stem_inception3_pool";
    pool(cur2, prv, cur2, "MAX", 3, 2);

    cur = "stem_inception3_concat";
    concat(cur, {cur1, cur2}, cur);
    return cur;
}

std::string inceptionA(std::string prv, int idx)
{
    sprintf(buf,"inception_a%d",idx);
    std::string header(buf);
    std::string cur1, cur2, cur3, cur4, prv1, prv3, prv4;

    cur1 = header + "_pool";
    pool(cur1, prv, cur1, "AVE", 3, 1, 1);

    prv1 = cur1;
    cur1 = header + "_conv1_1_1x1";
    convolution(cur1, prv1, cur1, 96, 0, 1, 1);
    norm(cur1);

    cur2 = header + "_conv2_1_1x1";
    convolution(cur2, prv, cur2, 96, 0, 1, 1);
    norm(cur2);

    cur3 = header + "_conv3_1_1x1";
    convolution(cur3, prv, cur3, 64, 0, 1, 1);
    norm(cur3);

    prv3 = cur3;
    cur3 = header + "_conv3_2_3x3";
    convolution(cur3, prv3, cur3, 96, 1, 3, 1);
    norm(cur3);

    cur4 = header + "_conv4_1_1x1";
    convolution(cur4, prv, cur4, 64, 0, 1, 1);
    norm(cur4);

    prv4 = cur4;
    cur4 = header + "_conv4_2_3x3";
    convolution(cur4, prv4, cur4, 96, 1, 3, 1);
    norm(cur4);

    prv4 = cur4;
    cur4 = header + "_conv4_3_3x3";
    convolution(cur4, prv4, cur4, 96, 1, 3, 1);
    norm(cur4);

    std::string cur = header + "_concat";
    concat(cur, {cur1, cur2, cur3, cur4}, cur);
    return cur;
}

std::string reductionA(std::string prv)
{
    std::string header = "reduction_a_";
    std::string cur1, cur2, cur3, prv3;

    cur1 = header + "pool";
    pool(cur1, prv, cur1, "MAX", 3, 2);

    cur2 = header + "conv2_1_3x3";
    convolution(cur2, prv, cur2, 384, 0, 3, 2);
    norm(cur2);

    cur3 = header + "conv3_1_1x1";
    convolution(cur3, prv, cur3, 192, 0, 1, 1);
    norm(cur3);
    
    prv3 = cur3;
    cur3 = header + "conv3_2_3x3";
    convolution(cur3, prv3, cur3, 224, 1, 3, 1);
    norm(cur3);

    prv3 = cur3;
    cur3 = header + "conv3_3_3x3";
    convolution(cur3, prv3, cur3, 256, 0, 3, 2);
    norm(cur3);

    std::string cur = header + "concat";
    concat(cur, {cur1, cur2, cur3}, cur);
    return cur;
}

int main()
{
    freopen("train_val.prototxt","w",stdout);
    create_data();
    std::string res = create_stem("data");
    for(int i=1;i<=4; i++)
        res = inceptionA(res, i);
    res = reductionA(res);
}
