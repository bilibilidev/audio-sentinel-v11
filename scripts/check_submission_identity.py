"""Scan the current source tree for private identity terms without publishing them.

Supply a local JSON list using --terms-file. Keep it and the report outside Git.
This is a text/NPZ-metadata search, not proof that images or narration are clear.
"""
import argparse,ast,io,json,pathlib,re,struct,subprocess,zipfile

def npz_strings(path):
    result=[]
    with zipfile.ZipFile(path) as archive:
        for name in archive.namelist():
            data=archive.read(name)
            if not data.startswith(b'\x93NUMPY'):continue
            major=data[6];offset=10 if major==1 else 12
            length=struct.unpack_from('<H' if major==1 else '<I',data,8)[0]
            header=ast.literal_eval(data[offset:offset+length].decode('latin1').strip())
            kind=str(header.get('descr',''));payload=data[offset+length:]
            if 'U' in kind:text=payload.decode('utf-32-be' if kind.startswith('>') else 'utf-32-le',errors='ignore')
            elif 'S' in kind or 'O' in kind:text=payload.decode('utf-8',errors='ignore')
            else:continue
            result.append((name,text.replace('\0',' ')))
    return result

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',type=pathlib.Path,default=pathlib.Path(__file__).resolve().parents[1])
    parser.add_argument('--terms-file',type=pathlib.Path,required=True)
    parser.add_argument('--output',type=pathlib.Path,required=True)
    parser.add_argument('--git',default='git')
    args=parser.parse_args();repo=args.repo.resolve();output=args.output.resolve()
    if output.is_relative_to(repo) and not output.is_relative_to(repo/'out'):
        parser.error('Keep the private report outside the repository or under ignored out/.')
    terms=json.loads(args.terms_file.read_text(encoding='utf-8-sig'))
    assert isinstance(terms,list) and terms and all(isinstance(s,str) and s.strip() for s in terms)
    private=[re.compile(re.escape(s),re.I) for s in terms]
    general=re.compile(r'大学|学院|学校|指导教师|指导老师|导师|校徽|\buniversity\b|\bcollege\b|\badvisor\b|\bsupervisor\b|\.edu\b',re.I)
    data=subprocess.check_output([args.git,'-C',str(repo),'ls-files','--cached','--others','--exclude-standard','-z'])
    names=sorted(set(data.decode('utf-8').split('\0'))-{''})
    hits=[];generic=[];text_files=0;arrays=0;images=[];skipped=[]
    def search(name,text,member=None):
        for line_no,line in enumerate(text.splitlines(),1):
            found=[terms[i] for i,p in enumerate(private) if p.search(line)]
            detail=dict(path=name,member=member,line=line_no,excerpt=line[:700])
            if found:hits.append(dict(detail,matched_terms=found))
            if general.search(line):generic.append(detail)
    for name in names:
        p=repo/name
        if not p.is_file():continue
        if p.suffix.lower() in {'.png','.jpg','.jpeg','.svg','.webp','.gif','.ico'}:images.append(name)
        if p.suffix=='.npz':
            for member,text in npz_strings(p):search(name,text,member);arrays+=1
            continue
        data=p.read_bytes()
        if b'\0' in data[:8192]:skipped.append(name);continue
        try:text=data.decode('utf-8-sig')
        except UnicodeDecodeError:
            try:text=data.decode('gb18030')
            except UnicodeDecodeError:skipped.append(name);continue
        text_files+=1;search(name,text)
    result=dict(scope='Current tracked and non-ignored working-tree code/text plus string/object metadata of NPZ arrays; Git history, ignored documents, video, visual logos and spoken audio not certified.',
                total_files=len(names),text_files=text_files,npz_string_members=arrays,exact_hits=hits,
                generic_mentions_for_manual_review=generic,image_assets_requiring_visual_review=images,binary_skipped=skipped,
                exact_match_count=len(hits),terms_stored_only_in_local_input=True,
                conclusion='No exact supplied term found' if not hits else 'Exact supplied terms found; inspect private report')
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(total_files=len(names),text_files=text_files,npz_string_members=arrays,exact_match_count=len(hits),
                          generic_review_count=len(generic),image_asset_count=len(images),private_report=str(output)),ensure_ascii=False))
    return 2 if hits else 0
if __name__=='__main__':raise SystemExit(main())
