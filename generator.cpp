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

int main()
{
    freopen("train_val.prototxt","w",stdout);
    create_data();
}
