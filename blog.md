---
layout: page
title: ""
---

Short posts on my side-projects and snippets from my studies, which I really enjoyed:

<ul>
{% for post in site.posts %}
    {% if post.categories contains "blog" %}
        <li><a href="{{ post.url }}">{{ post.title }}</a></li>
    {% endif %}
{% endfor %}
</ul>