import torch
import torch.nn.functional as F
from loss_math import (teacher_topk, opd_token_losses, text_denominators,
                       image_mean, image_group_table, image_token_losses)


def test_distinct_sft_and_opd_diffusion_normalizers():
    assert text_denominators(100, 20, opd=False) == (100, 20)
    assert text_denominators(100, 20, opd=True) == (100, 50.05)
    assert text_denominators(100, 0, opd=False) == (100, 1)
    assert image_mean(torch.tensor(60.), 20) == 3


def test_teacher_legal_support_keeps_full_normalizer():
    logits = torch.zeros(2, 256)
    logits[:, 100] = 10
    logp, ids = teacher_topk(logits)
    assert not (ids == 100).any()
    assert torch.allclose(logp.exp().sum(-1), torch.full((2,),64/(255+torch.exp(torch.tensor(10.)))))


def test_consistency_identity_and_stop_gradient():
    torch.manual_seed(81)
    ar=torch.randn(3,256,requires_grad=True)
    df=ar.detach().clone().requires_grad_()
    teacher=torch.randn(3,256,requires_grad=True)
    tp,ti=teacher_topk(teacher)
    ar_loss, diff_loss=opd_token_losses(ar,df,tp,ti)
    torch.testing.assert_close(diff_loss,torch.zeros(3),atol=1e-6,rtol=0)
    assert torch.autograd.grad(diff_loss.sum(),ar,allow_unused=True,retain_graph=True)[0] is None
    assert torch.autograd.grad(ar_loss.sum(),teacher,allow_unused=True,retain_graph=True)[0] is None
    assert torch.autograd.grad(ar_loss.sum(),ar)[0].abs().sum()>0


def test_grouped_image_ce_matches_group_probability_and_gradients():
    torch.manual_seed(7)
    logits=torch.randn(3,16,requires_grad=True)
    groups=torch.arange(16)//4
    members,valid=image_group_table(groups)
    targets=torch.tensor([0,6,15])
    exact,grouped=image_token_losses(logits,targets,groups,members,valid)
    expected=-torch.stack([logits.softmax(-1)[i,groups==groups[t]].sum().log() for i,t in enumerate(targets)])
    torch.testing.assert_close(exact,F.cross_entropy(logits,targets,reduction='none'))
    torch.testing.assert_close(grouped,expected)
    ag=torch.autograd.grad((exact+.2*grouped).sum(),logits,retain_graph=True)[0]
    eg=torch.autograd.grad((F.cross_entropy(logits,targets,reduction='sum')+.2*expected.sum()),logits)[0]
    torch.testing.assert_close(ag,eg)


def test_sft_keeps_student_added_vocabulary():
    from types import SimpleNamespace
    from objectives import Student
    class Encoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens=torch.nn.Embedding(132101,4)
        def forward(self,inputs_embeds,**kwargs):
            return SimpleNamespace(last_hidden_state=inputs_embeds)
    encoder=Encoder()
    model=SimpleNamespace(device='cpu',encoder=encoder,diffusion_head=torch.nn.Linear(4,132101,bias=False))
    ids=torch.tensor([[9,131099,11,11]])
    labels=torch.tensor([[-100,131099,11,11]])
    student=SimpleNamespace(model=model,options={'block_size':2},
        sft_tokens=lambda _: (ids,encoder.embed_tokens(ids),labels))
    loss,metrics=Student.language_loss(student,None)
    assert torch.isfinite(loss)
    loss.backward()
    assert model.diffusion_head.weight.grad[131099].abs().sum()>0
