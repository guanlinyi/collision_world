// Explicit URL ownership; resources never depend on Referer or another tab's case.
export function casePrefix(pathname){
 if(pathname==='/'||pathname==='/index.html')return '';
 const match=/^\/cases\/(07|11)\/(?:index\.html)?$/.exec(pathname);
 if(!match)throw new Error('未知回放案例路径');
 return '/cases/'+match[1];
}
export function caseDataUrl(url,pathname=globalThis.location.pathname){
 return url.startsWith('/data/')?casePrefix(pathname)+url:url;
}
export function initialView(hash){
 const params=new URLSearchParams(hash.replace(/^#/,''));
 if([...params.keys()].some(k=>!['frame','car','method'].includes(k)))return {};
 const result={};
 if(params.has('frame')&&/^(?:[0-9]|[1-7][0-9])$/.test(params.get('frame')))result.frame=Number(params.get('frame'));
 if(params.has('car')&&/^[01]$/.test(params.get('car')))result.car=Number(params.get('car'));
 if(params.has('method')&&/^[a-z0-9_]{1,80}$/.test(params.get('method')))result.method=params.get('method');
 return result;
}
