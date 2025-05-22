#include<bits/stdc++.h>
using namespace std;
int main()
{
    int t ;
    cin>>t;
    while(t--)
    {
        int n;
        cin>>n;
        vector<string>ans;
        queue<string> q;
        q.push("1");
        while(!q.empty() && ans.size()<n)
        {
            string tmp=q.front();
            string a= tmp+"0";
            string b= tmp+"1";
            q.pop();
            q.push(a);
            q.push(b);
            ans.push_back(tmp);
        }
        for(auto x:ans) cout<<x<<" ";
        cout<<"\n";
    }
}