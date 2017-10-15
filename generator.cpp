#include<stdio.h>
#include<string>

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

//TODO: handle hyperparams outside
void convolution(std::string name, std::string bot, std::string top,
        int output, int pad, int kernel, int stride)
{
    print("layer {");
    sprintf(buf,"name: \"%s\"",name.c_str());
    print("type: \"Convolution\"");
    sprintf(buf,"bottom: \"%s\"",bot.c_str());
    print(buf);
    sprintf(buf,"top: \"%s\"",top.c_str());
    print(buf);

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
    convolution("stem_conv1_3x3", "data", "stem_conv1_3x3",32,0,3,2);
}

int main()
{
    freopen("train_val.prototxt","w",stdout);
    create_data();
    create_stem();
}
